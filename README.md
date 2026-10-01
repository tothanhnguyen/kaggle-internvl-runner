# kaggle-internvl-runner
# kaggle-internvl-runner

## Cấu hình chung cho ba model

Gửi nhóm [configs/experiment.yaml](configs/experiment.yaml) và [common/prompt.txt](common/prompt.txt) từ cùng phiên bản repo. Không gửi kèm ảnh, CSV case-level, model weights hay predictions lên Git.

Phần dùng chung: split/CSV, thứ tự `left_cc → left_mlo → right_cc → right_mlo`, bốn ảnh/case, prompt A–D, `do_sample: false`, `max_new_tokens: 4`, `num_beams: 1` và evaluator. Không thay protocol riêng cho từng model.

| Block | Dtype theo checkpoint | Xử lý ảnh riêng |
|---|---|---|
| `models.qwen` | `bfloat16` | Native dynamic resolution; `processor.min_pixels/max_pixels` là ngân sách pilot 256–1280 visual tokens/view. |
| `models.llava` | `float16` | CLIP resize/center crop 336×336; patch 14; template Phi-3. |
| `models.internvl` | `bfloat16` | Tile 448px, tối đa 4 tile/view trước thumbnail; bốn image placeholders. |

Cả ba có revision đã ghim, device và quantization (`none`). Đây là so sánh native pipelines, **không phải** cùng precision, kích thước ảnh sau xử lý hay số visual tokens. Chốt và ghi lại khác biệt sau pilot; nếu muốn cùng dtype thì phải thử/chốt với cả nhóm trước test.

### Nối runner Qwen/LLaVA

Repo hiện chỉ có runner InternVL. Các trường Qwen/LLaVA không tự có hiệu lực trong script của hai bạn nếu script chưa đọc chúng:

1. Chọn block `models.qwen` hoặc `models.llava`. Nạp lần lượt bằng `Qwen2VLForConditionalGeneration` hoặc `LlavaForConditionalGeneration`; dùng `checkpoint`, `revision`, `trust_remote_code`, và chuyển `dtype` sang kiểu `torch` tương ứng. `device: auto` của project là CUDA nếu có, không có thì báo lỗi; không tự CPU-offload hay quantize.
2. Dùng `AutoProcessor.from_pretrained(checkpoint, revision=revision, trust_remote_code=trust_remote_code, **processor_settings)` với `processor_settings = model_config["processor"]`.
3. Qwen: một user message chứa bốn item image đúng thứ tự rồi một item text là prompt chung; gọi `processor.apply_chat_template(..., tokenize=False, add_generation_prompt=True)`.
4. LLaVA: dùng `model_config["prompt_template"]`; `{images}` là bốn dòng `processor.image_token`, `{prompt}` là prompt chung. `patch_size`, `vision_feature_select_strategy`, `num_additional_image_tokens` trong config giúp processor mở rộng đúng 576 tokens/view. Không dùng template Vicuna/InternVL.
5. Truyền cùng bốn ảnh RGB và một prompt vào processor; dùng `generation` chung. Không truncate mất image tokens; LLaVA cần giữ tổng input + output trong context 4096. Decode chỉ phần token sinh mới.
6. Parser chung: trim/uppercase rồi chỉ chấp nhận đúng một ký tự A/B/C/D; khác thì `INVALID`. Giữ raw output, không bỏ case lỗi; dùng evaluator chung.

LLaVA cần pilot bốn ảnh trên validation: [Transformers cảnh báo multi-image chưa được huấn luyện rõ ràng](https://huggingface.co/docs/transformers/v4.52.3/en/model_doc/llava#usage-tips). Nếu không đáp ứng, cả nhóm phải thống nhất protocol thay thế, không tự đổi riêng model này sang một ảnh.

### Phạm vi thay đổi và kiểm tra

- Đợt bổ sung config này chỉ thêm runtime/processor/template Qwen/LLaVA; không đổi protocol chung, prompt, block InternVL hay đường dẫn output.
- Đã kiểm tra `--check-config` của InternVL và processor thật của Qwen/LLaVA trên bốn ảnh giả lập với Transformers 4.52.3. Chưa nạp trọng số hay chạy inference ba model.
- `--check-config` không kiểm tra runner Qwen/LLaVA bên ngoài repo. Người chạy cần ghi package versions, hardware, dtype và config runtime thực tế.
- Model/processor theo nguồn: [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/tree/895c3a49bc3fa70a340399125c650a463535e71c), [LLaVA-Phi3](https://huggingface.co/xtuner/llava-phi-3-mini-hf/tree/218fa56e23d2b894dd13f2c4ecf4b90843b12b39). Giới hạn pixel Qwen là lựa chọn pilot theo ví dụ model card, không phải default upstream.
