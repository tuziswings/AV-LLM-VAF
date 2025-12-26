import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from AttnAdapter import AttnAdapter, SegmentInfo, attention_compute, saliency_compute

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
VIDEOLLAMA_ROOT = WORKSPACE_ROOT / "AVCD"
if str(VIDEOLLAMA_ROOT) not in sys.path:
    sys.path.append(str(VIDEOLLAMA_ROOT))

from videollama2 import model_init  # noqa: E402
from videollama2.constants import DEFAULT_VIDEO_TOKEN  # noqa: E402
from videollama2.mm_utils import tokenizer_multimodal_token  # noqa: E402
from videollama2.utils import disable_torch_init  # noqa: E402

VIDEO_TOKEN_LEN = 676


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def resolve_video_filename(sample, args):
    if "video" in sample:
        filename = sample["video"]
    elif "image" in sample:
        filename = sample["image"]
    elif "video_id" in sample:
        vid = str(sample["video_id"])
        if args.video_id_width is not None and vid.isdigit():
            vid = vid.zfill(args.video_id_width)
        filename = vid + args.video_suffix
    else:
        raise KeyError("Sample does not contain video/image identifier.")

    if os.path.isabs(filename):
        return filename
    return os.path.join(args.video_folder, filename)


def encode_label(tokenizer, label_text):
    token_ids = tokenizer(label_text, add_special_tokens=False).input_ids
    if not token_ids:
        token_ids = tokenizer(label_text, add_special_tokens=True).input_ids
    if not token_ids:
        raise ValueError(f"Failed to tokenize label: {label_text}")
    return token_ids[0]


def build_messages(question, modal_token, model_type):
    user_content = modal_token + "\n" + question
    messages = []
    if model_type in {"videollama2", "videollama2_mistral", "videollama2_mixtral", "videollama2_qwen2"}:
        system_prompt = (
            "<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, "
            "while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, "
            "dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n"
            "If a question does not make any sense, or is not factually coherent, explain why instead of answering "
            "something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>"
        )
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    return messages


def prepare_segment_info(start_point, last_point, seq_len, video_token_len, visual_len=None, audio_len=None):
    if start_point is None or last_point is None:
        return None
    mm_tokens = max(0, last_point - start_point)
    if visual_len is None or audio_len is None:
        visual = min(video_token_len, mm_tokens)
        audio = max(0, mm_tokens - visual)
    else:
        visual = visual_len
        audio = audio_len
    text = max(0, seq_len - last_point)
    return SegmentInfo(system=start_point, visual=visual, audio=audio, text=text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--video-folder", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--video-suffix", type=str, default=".mp4")
    parser.add_argument("--video-id-width", type=int, default=5)
    parser.add_argument("--video-token-len", type=int, default=VIDEO_TOKEN_LEN)
    parser.add_argument("--num-frames", type=int, default=None, 
                        help="Number of video frames to process (default: use model config, typically 8). "
                             "Reduce to 4 to save memory (~75% reduction in attention matrix size).")
    args = parser.parse_args()

    disable_torch_init()
    if args.device.startswith("cuda"):
        torch.cuda.set_device(int(args.device.split(":")[1]))

    model, processor, tokenizer = model_init(args.model_path, device=args.device)
    model.eval()
    device = torch.device(args.device)
    model_dtype = next(model.parameters()).dtype

    # Override num_frames if specified (to reduce memory usage)
    if args.num_frames is not None:
        from functools import partial
        from videollama2.mm_utils import process_video
        # Get the underlying processor from the partial function
        original_video_processor = processor["video"]
        underlying_processor = original_video_processor.keywords.get("processor")
        if underlying_processor is None:
            # Fallback: get from model's vision tower processor
            underlying_processor = model.get_vision_tower().image_processor
        # Create new processor with reduced num_frames
        processor["video"] = partial(process_video, processor=underlying_processor, 
                                     aspect_ratio=None, num_frames=args.num_frames, va=True)
        default_frames = model.config.num_frames if hasattr(model.config, 'num_frames') else 8
        print(f"Using reduced num_frames={args.num_frames} (default: {default_frames}) "
              f"for memory efficiency. Attention matrix size will be ~{(args.num_frames/default_frames)**2:.1%} of original.")

    model_type = getattr(model.config, "model_type", "")
    for layer_idx, layer in enumerate(model.model.layers):
        attn_adap = AttnAdapter(layer.self_attn.config, layer_idx=layer_idx)
        attn_adap.load_state_dict(layer.self_attn.state_dict())
        attn_adap = attn_adap.to(device=device, dtype=model_dtype)
        layer.self_attn = attn_adap

    with open(os.path.expanduser(args.question_file), "r") as f:
        questions = [json.loads(q) for q in f]

    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    if args.sample_size > 0 and len(questions) > args.sample_size:
        questions = random.sample(questions, args.sample_size)

    results = []
    modal_token = DEFAULT_VIDEO_TOKEN

    for sample in tqdm(questions):
        question = sample["text"] + " Please just answer yes or no."
        label_text = sample.get("label")
        if label_text is None:
            continue
        label_id = encode_label(tokenizer, label_text)
        label = torch.tensor(label_id, dtype=torch.int64, device=device)

        video_path = resolve_video_filename(sample, args)
        av_inputs = processor["video"](video_path, va=True)
        if not isinstance(av_inputs, dict):
            raise ValueError("Expected video processor to return dict with 'video' and 'audio'.")
        av_inputs = {k: v.to(device=device, dtype=model_dtype) for k, v in av_inputs.items()}
        multimodal_inputs = [(av_inputs, "video")]

        messages = build_messages(question, modal_token, model.config.model_type)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer_multimodal_token(prompt, tokenizer, modal_token, return_tensors='pt').unsqueeze(0).to(device)
        attention_mask = input_ids.ne(tokenizer.pad_token_id).long().to(device)

        (
            _,
            attn_mask,
            _,
            inputs_embeds,
            _,
            start_point,
            last_point
        ) = model.prepare_inputs_labels_for_multimodal(
            input_ids.clone(),
            attention_mask.clone(),
            None,
            None,
            multimodal_inputs,
        )

        visual_tokens = getattr(model, "last_visual_token_len", None)
        audio_tokens = getattr(model, "last_audio_token_len", None)

        segment_info = prepare_segment_info(
            start_point,
            last_point,
            inputs_embeds.shape[1],
            args.video_token_len,
            visual_len=visual_tokens,
            audio_len=audio_tokens,
        )
        if segment_info is None or segment_info.visual == 0:
            continue
        
        # Debug: Print token count verification (only for first sample)
        if len(results) == 0:
            print("\n" + "="*80)
            print("DEBUG: Token Indexing Verification (First Sample)")
            print("="*80)
            print(f"\nFrom prepare_inputs_labels_for_multimodal:")
            print(f"  start_point: {start_point}")
            print(f"  last_point: {last_point}")
            print(f"  inputs_embeds.shape[1]: {inputs_embeds.shape[1]}")
            print(f"\nFrom model attributes:")
            print(f"  last_visual_token_len: {visual_tokens}")
            print(f"  last_audio_token_len: {audio_tokens}")
            if visual_tokens is not None and audio_tokens is not None:
                print(f"  visual + audio: {visual_tokens + audio_tokens}")
                mm_tokens = last_point - start_point if last_point and start_point else 0
                print(f"  mm_tokens (last_point - start_point): {mm_tokens}")
                if visual_tokens + audio_tokens != mm_tokens:
                    print(f"  ⚠️  WARNING: visual + audio != mm_tokens")
                else:
                    print(f"  ✓ visual + audio == mm_tokens")
            print(f"\nSegmentInfo:")
            print(f"  system: {segment_info.system}")
            print(f"  visual: {segment_info.visual}")
            print(f"  audio: {segment_info.audio}")
            print(f"  text: {segment_info.text}")
            total = segment_info.system + segment_info.visual + segment_info.audio + segment_info.text
            print(f"  Total: {total}")
            if total != inputs_embeds.shape[1]:
                print(f"  ⚠️  WARNING: Total != seq_len")
            else:
                print(f"  ✓ Total == seq_len")
            from AttnAdapter import _segment_offsets
            sys_end, vis_end, aud_end, txt_end = _segment_offsets(segment_info)
            print(f"\nSegment Offsets:")
            print(f"  sys_end: {sys_end} (indices 0 to {sys_end-1})")
            print(f"  vis_end: {vis_end} (indices {sys_end} to {vis_end-1}, length={vis_end-sys_end})")
            print(f"  aud_end: {aud_end} (indices {vis_end} to {aud_end-1}, length={aud_end-vis_end})")
            print(f"  txt_end: {txt_end} (indices {aud_end} to {txt_end-1}, length={txt_end-aud_end})")
            if vis_end - sys_end != segment_info.visual:
                print(f"  ⚠️  WARNING: vis_end - sys_end != visual")
            if aud_end - vis_end != segment_info.audio:
                print(f"  ⚠️  WARNING: aud_end - vis_end != audio")
            if txt_end - aud_end != segment_info.text:
                print(f"  ⚠️  WARNING: txt_end - aud_end != text")
            if txt_end != inputs_embeds.shape[1]:
                print(f"  ⚠️  WARNING: txt_end != seq_len")
            print("="*80 + "\n")

        model.zero_grad(set_to_none=True)
        
        with torch.cuda.amp.autocast(enabled=False):
            model_outputs = model(
                attention_mask=attn_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=True,
                return_dict=True,
            )
            
            if isinstance(model_outputs, tuple) and len(model_outputs) == 3:
                outputs, avg_dominance, threshold = model_outputs
            else:
                outputs = model_outputs

            logits = outputs.logits[:, -1, :].squeeze(0)
            loss = F.cross_entropy(logits, label)
            
            del logits
            torch.cuda.empty_cache()
            
            loss.backward()
            
            del loss

            img_flow, layers_attn = [], []
            for idx_layer, layer in enumerate(model.model.layers):
                attn_grad = layer.self_attn.attn_map.grad.detach().cpu()
                attn_score = outputs.attentions[idx_layer].detach().cpu()
                
                layer.self_attn.attn_map.grad = None
                
                saliency = torch.abs(attn_grad * attn_score)

                img_saliency = saliency_compute(saliency, segment_info)
                attn_props = attention_compute(attn_score, segment_info)

                img_flow.append(img_saliency)
                layers_attn.append(attn_props)
                
                del attn_grad, attn_score, saliency
                
                if (idx_layer + 1) % 5 == 0:
                    torch.cuda.empty_cache()

            results.append((img_flow, layers_attn))
            
            del outputs
            for layer in model.model.layers:
                if hasattr(layer.self_attn, 'attn_map') and layer.self_attn.attn_map.grad is not None:
                    layer.self_attn.attn_map.grad = None
            
            del av_inputs, multimodal_inputs, inputs_embeds, attn_mask, input_ids, attention_mask
            torch.cuda.empty_cache()

    torch.save(results, args.answers_file)


if __name__ == "__main__":
    main()