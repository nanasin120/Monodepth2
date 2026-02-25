# Monodepth2
본 프로젝트는 [Digging Into Self-Supervised Monocular Depth Estimation]논문을 바탕으로 했음을 알립니다.

# 1. 프로젝트 소개
본 프로젝트는 Monodepth2를 이용해 Unity가상환경을 통해 구한 데이터에서 깊이 정보를 자기-지도 학습하는 프로젝트입니다.
<img width="1280" height="192" alt="tmpmheujs5n" src="https://github.com/user-attachments/assets/c79f003c-d4ca-44a8-a3aa-344f073171d6" />
<img width="1280" height="192" alt="tmpuoeicoae" src="https://github.com/user-attachments/assets/bd559019-5f0e-40aa-bbde-e769d53a5e2b" />
위 사진은 유니티 세상에서 찍은 데이터로 추정한 깊이와 실제 현실 세상에서 찍은 데이터로 추정한 깊이입니다.

## Tech Stack
* Environment: Unity (데이터 생성)
* Deep Learning : Pytorch (Monodepth2 모델 설계 및 학습)
* Visualization : PIL (이미지 출력)

# 2. 데이터 생성
유니티 환경을 통해 데이터를 생성했습니다.
| 데이터 | 실제 모습 |
|---|---|
| 카메라 1대를 통한 정면 이미지 | ![frame_000000_cam_0](https://github.com/user-attachments/assets/4ef61702-74e2-4cdb-9716-a446dc344db6) |
| 내부 행렬 | csv형태로 저장 |

총 1238개의 데이터를 사용했습니다.

# 3. 모델 구현
| 이름 | 기능 |
|---|---|
| depthNetwork | 이미지로부터 깊이를 추정한다. | 
| poseNetwork | 두개의 이미지를 통해 카메라의 회전과 이동을 추정한다. |
| projector | 회전과 이동, 깊이를 통해 가상의 이미지를 투영한다. |

# 4. 학습
설정한 하이퍼파라미터들입니다.
| 항목 | 설정값 | 비고 |
| --- | --- | --- |
| Epoch | 300 | 최종 학습 횟수 |
| Batch Size | 8 | 너무 많지도 작지도 않은 사이즈였습니다. |
| Learning rate | 0.0001 | 초기 학습률 |
| Optimizer | Adam | 논문의 내용을 따랐습니다. |
| Image Size | 192 x 640 | 논문의 내용을 따랐습니다. |

손실함수는 논문에 따라 Minimum_Reprojection_Loss + 0.001 * Smooth_Loss로 구성했습니다.

# 5. 결과
<img width="1331" height="227" alt="스크린샷 2026-02-25 005533" src="https://github.com/user-attachments/assets/4e816ab8-3ef9-4cc8-9307-d83294c6e11b" />
<img width="1304" height="226" alt="image" src="https://github.com/user-attachments/assets/2e4bc8ac-6dc0-4520-bedf-3885972d3625" />
<img width="1304" height="217" alt="image" src="https://github.com/user-attachments/assets/cebff570-262e-4652-a176-587b884b5f0c" />

어느정도 경계선도 잡고 원거리 지평선도 잡은 것으로 보입니다. 하지만 멀리있는 건물이나 나무와 같은 구조물은 아직 잘 잡지 못하는것으로 보입니다.
