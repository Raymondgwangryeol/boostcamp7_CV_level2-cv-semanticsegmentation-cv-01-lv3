# 🦴 Hand Bone Image Segmentation
>  **Boostcamp AI Tech 7기 | Computer Vision Track**  
>  
> 손 뼈 X-ray 영상을 기반으로 다양한 뼈 구조를 정밀하게 분할하는 Semantic Segmentation 프로젝트입니다.

## 프로젝트 소개

뼈는 우리 몸의 구조와 기능에 핵심적인 역할을 하기 때문에, 정확한 뼈 분할은 의료 진단 및 수술 계획 수립 등 다양한 의료 분야에서 매우 중요합니다.

본 프로젝트는 X-ray 이미지를 활용한 뼈 분할 모델을 개발하여, 다음과 같은 의료 응용에 기여하고자 했습니다:

- **질병 진단**: 뼈의 위치 이상, 변형, 골절 등을 정밀하게 파악하여 정확한 진단을 돕습니다.
- **수술 계획**: 뼈 구조를 기반으로 수술 접근 방식과 필요한 재료 등을 설계합니다.
- **의료 장비 제작**: 인공 관절, 임플란트 등 맞춤형 의료 장비를 설계하는 데 필요한 구조 정보를 제공합니다.
- **의료 교육**: 의학 교육 및 시뮬레이션에서 실제와 유사한 데이터를 제공하여 이해도를 높입니다.

딥러닝 기반의 segmentation 기술을 활용해 29개 손 뼈 클래스를 픽셀 단위로 정밀 분할합니다.



## 사용 기술

- **Semantic Segmentation**: MMSegmentation v3
- **학습 프레임워크**: `PyTorch`
- **데이터 포맷**: 이미지 (X-ray) + json 기반 segmentation annotation
- **실험 관리**: `Weights & Biases`, `Notion`, `Git branch 전략`
- **기타**: `Albumentations`, `OpenCV`, `timm`



## 데이터

- **Input**: 손 뼈 X-ray 이미지
- **Annotation**: JSON 형식의 segmentation mask (총 29개 클래스)
- **Output**: 모델은 각 클래스에 대한 확률 맵을 생성하고, 이를 통해 픽셀별 클래스를 예측
- **Submission**: 예측된 마스크는 `.csv`로 제출


## 기여한 부분
#### Curriculum Learning 실험 및 적용

- **학습 난이도 기반 데이터 구성**: 손등뼈(finger-1)를 제외한 쉬운 샘플로 먼저 학습한 뒤 전체 손 뼈를 학습하는 curriculum learning 전략 설계
- **Self-paced learning 구현**: loss 값 기반으로 쉬운 샘플을 우선적으로 학습하도록 vi 가중치 도입
- **학습 전략 효과 분석**: 각 단계에서의 dice score 변화 분석 및 시각화로 curriculum learning의 효과 입증

#### 실험 지연 상황 속 주도적인 문제 해결 시도

- 초기 base 코드가 정상 작동하지 않는 문제가 있었으며, 문제 해결을 위해 원인 파악에 시간을 투자
- 원인 분석을 우선적으로 진행하며 문제 해결을 시도했으며, 타인에게 무작정 의존하기보다는 스스로 파악하고자 하는 책임감 있는 태도를 유지
- 그러나 일정 측면에서 빠른 피드백 루틴의 중요성을 체감, 이후 프로젝트에서는 더 신속한 커뮤니케이션과 협업을 실천

#### 자원 제약 속 실험 마무리

- 대회 종료 직전까지 실험을 지속하며 최대한의 성능 확인 및 비교 실험 수행
- 사전 지식 기반 curriculum과 loss 기반 self-paced 학습 전략을 비교한 결과, self-paced 방식이 더 효과적임을 실험적으로 입증
- 시간과 리소스 제약 속에서도 끝까지 실험을 마무리하며, 실제 성능 개선과 함께 전략의 가능성을 확인


### 실험 결과 요약

- 손등뼈(finger-1)를 제외한 쉬운 샘플로 먼저 학습한 모델은 빠르게 loss가 감소하며 학습됨
- 이후 전체 손 뼈를 학습 시, base code보다 모든 클래스의 dice score가 높았음
- Self-paced 방식 curriculum learning은 **Validation Dice Score: 0.9657**로, base code의 **0.9609**보다 우수한 성능 달성

> Curriculum learning은 적절히 설계된다면 segmentation 성능을 향상시킬 수 있는 효과적인 접근 방식임을 확인함

## Project Structure

```
** project structure **

level2-cv-semanticsegmentation-cv-01-lv3/
│
├── data/
│   ├── train/
│   │    ├── DCM / ID ###
│   │    └── outputs_json / ID ###
│   │
│   └── test /
│        └── DCM / ID ###
├──utils/
│   ├── dataset.py
│   ├── method.py
│   ├── augmentation.py
│   ├── handrotation.py
│   ├── hard_voting.py
│   ├── trainer.py
│   └── visualization.py
│
├── inference.py
├── train.py
├── requirements.txt
└── README.md
```

### 실험 환경 구성
- 초기 실험 환경: PyTorch + Segmentation Models Pytorch (SMP)
- 주요 실험 및 성능 보고: MMSegmentation v3 기반 커스텀 구성

## Model Architecture
본 프로젝트는 **MMSegmentation v3**를 기반으로 진행되었습니다.

### 모델 구성
#### DeepLabV3Plus
- **Main Framework**: DeepLabV3Plus
- **Backbone**: SwinTransformer(Swin-Tiny)
- **Optimizer**: AdamW
- **Input size**: (512, 512)

## Training Script
본 프로젝트에서는 mmsegmentation3의 기본 train.py를 기반으로, 실험에 맞게 수정한 커스텀 스크립트를 사용했습니다.

### 주요 특징
- 학습 스크립트는 argparse를 활용해 다양한 설정을 인자로 받아 유연하게 구성할 수 있도록 설계하였습니다.
- 모델 구조, 데이터 경로, 학습 하이퍼파라미터, WandB 로깅 등 주요 설정을 command-line 인자로 받아 실행 가능하게 구성하였습니다.
- 학습 스크립트에서 parser를 활용해 설정값을 외부에서 제어할 수 있도록 구현함으로써, 실험 반복에 유리한 구조를 설계하였습니다.

   
