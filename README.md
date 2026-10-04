# Mini-CompressKV — 단계별 학습 구현

첨부 번역 정리의 핵심 아이디어를 직접 구현한 교육용 프로젝트입니다.
공식 구현을 복사한 프로젝트나 논문 성능 재현 결과가 아닙니다.

## 바로 실행: 모델 없이 알고리즘 확인

```bash
python -m pip install "numpy>=1.26,<3"
python -m unittest -v test_core.py
python demo.py
```

`results/synthetic.json`은 합성 텐서의 결과이며 LLM 정확도가 아닙니다.

## 실제 Qwen 모델 비교 (실행 환경에서 검증 필요)

Python 3.10~3.12 권장. 가상환경에서 requirements.txt를 설치하세요.
CUDA 환경은 해당 GPU에 맞는 PyTorch를 먼저 설치하세요.

```bash
python -m pip install -r requirements.txt
python run_qwen.py --method full --device cpu
python run_qwen.py --method recent --budget 128 --device cpu
python run_qwen.py --method attention --budget 128 --device cpu
```

GPU에서는 `--device cuda`, Mac에서는 `--device mps`를 사용할 수 있습니다.
모델은 최초 실행 때 다운로드합니다. 현재 제작 환경에는 torch가 없어
실제 모델 러너는 실행 검증하지 못했습니다. 작은 Qwen의 검색 능력이
낮을 수 있으므로 Full KV 정답 여부부터 확인해야 합니다.

## 파일과 학습 순서

1. `core.py`의 `srh_scores`: 정답 위치에 대한 attention 질량을 합산.
2. `token_scores`: 마지막 8개 query → 토큰 평균 pooling → 선택 헤드 평균.
3. `select_indices`: 최근 토큰 보호 + 과거 중요 토큰 top-k + 원래 순서 복원.
4. `compress_kv`: 레이어 내 모든 KV 헤드에 공통 위치 집합 적용.
5. `gqa_output` / `relative_error`: 동일 Q의 full/압축 출력을 투영 후 비교.
6. `allocate_budget`: 최소·최대 제약, 정수 반올림, 정확한 전체 예산 보존.
7. `run_qwen.py`: 실제 모델의 Full/최근/모든-head attention 기준선 비교.

KV 모양: [batch, KV_heads, sequence, head_dim].
Attention 모양: [batch, query_heads, query_positions, key_positions].
GQA에서는 query_heads와 KV_heads가 다를 수 있습니다.

## 정리본을 코드로 옮길 때 결정한 사항

- 정답 스팬의 **위치**와 생성된 **토큰 ID**를 분리합니다.
  SRH gating은 정답 토큰 ID 집합 포함 여부로 단순화했습니다.
  중복된 일반 단어도 맞았다고 집계할 수 있어 공식 방식과 추가 대조가 필요합니다.
- pooling 종류는 정리본에서 확정할 수 없어 **평균 pooling**, zero padding을
  명시적으로 사용했습니다. 공식 코드와 동일하다는 주장은 하지 않습니다.
- 최근 window 보호는 이 교육 구현의 명시적 설계 선택입니다.
- 생성 ID와 attention은 그 ID를 예측한 forward 시점에 맞춰야 합니다.
- 레이어 error는 동일 query에서 KV만 바꿔 비교하는 국소 실험입니다.
  `gqa_output` 입력 Q/K에는 모델의 RoPE가 이미 적용되어 있어야 합니다.
- 실제 러너는 absolute position을 유지하고 legacy tuple cache API를 사용하도록
  Transformers 4.44.2에 고정했습니다. 버전 업그레이드는 별도 검증이 필요합니다.
- 프레필 후 한 번 압축하며 생성 KV는 계속 추가됩니다. 고정 총용량 보장은 아닙니다.
- 전체 attention을 만드는 eager 방식은 O(sequence²) 메모리가 필요합니다.
  먼저 짧은 문맥으로 진행하세요. 이 코드는 FlashAttention 속도 재현용이 아닙니다.
- 기록된 KV 바이트는 cache 텐서 크기이며 전체 GPU peak memory가 아닙니다.
  압축 선택 중 full cache도 존재하므로 peak가 같은 비율로 줄지는 않습니다.

## 다음 실험: 실제 SRH 보정 통합

현재 SRH/오차/예산 알고리즘은 합성 입력으로 검증했고 실제 모델 러너에는
연결하지 않았습니다. 연결 없이 `attention`을 CompressKV라고 부르면 안 됩니다.

1. 학습용 보정 문서와 평가 문서를 분리하고 정답 char span을 token span으로 매핑.
2. full-cache 생성에서 생성 토큰·그 시점 attention을 저장하고 SRH 점수 누적.
3. 레이어당 top-4 query head 선택. 점수가 모두 0인 레이어의 fallback 명시.
4. 동일 Q/K/V와 o_proj로 여러 보정 decode step의 상대 오차를 합산.
5. 레이어 최소 32, 최대 min(prompt_length, 3*평균예산)로 예산 계산.
6. head/예산 설정을 모델·토크나이저·보정 데이터 버전과 함께 저장.
7. 평가에서는 정답 스팬을 제공하지 않고 보정 결과만 사용해 압축.
8. 다수 문맥 길이·needle 위치·seed에서 Full/Recent/Random/Attention/SRH/
   SRH+Adaptive 비교. warm-up 후 여러 번 측정하고 평균/분산 보고.

논문 보고 수치를 이 소형 모델의 목표 정확도로 사용하지 마세요.

## 출처

- 사용자 제공 CompressKV 번역 정리
- 공식 저장소: https://github.com/TUDa-HWAI/CompressKV
- 논문 링크: https://arxiv.org/abs/2606.24467
