# Urban Heat Island Intensity Prediction with Multi-Stream CNN and Attention U-Net

## Overview
This project develops a deep learning framework to predict Urban Heat Island Intensity (UHII) in Pune, India, using a multi-stream convolutional neural network (CNN) with attention-based fusion. By integrating three geospatial data modalities—Land Surface Temperature (LST), PM₂.₅ particulate matter, and high-resolution land-cover maps—our model achieves a Mean Squared Error (MSE) of 0.1104 on held-out test data, outperforming baseline models (MSE=0.185). The framework leverages Google Earth Engine (GEE) for scalable data processing and provides interpretable attention maps to guide urban planning and climate resilience strategies.

### Key Features
- **Multi-Modal Inputs**: Processes Terra MODIS LST (MOD11A2), GLC_FCS30D land-cover, and satellite-derived PM₂.₅ datasets.
- **Model Architecture**: Coordinated encoders with CoordConv and CBAM attention, fused via an Attention U-Net for end-to-end UHII prediction.
- **Humanitarian Impact**: Identifies heat risk drivers (e.g., impervious surfaces, pollution) to inform urban greening and emission control policies.
- **Interactive Pipeline**: Jupyter notebook with widgets for temporal (2003–2020) and seasonal data exploration.

## Dataset
- **Sources**:
  - **MOD11A2 LST**: 8-day, 1 km resolution [NASA LP DAAC, 2025].
  - **GLC_FCS30D Land Cover**: Annual, 30 m resolution, 2000–2022 [GEE Community Catalog, 2025].
  - **PM₂.₅**: Monthly, 0.01° resolution [GEE Community Catalog, 2025].
  - **UHII Labels**: Derived from urban-rural LST differences [GEE Community Catalog, 2025].
- **Preprocessing**:
  - Clipped to Pune’s administrative boundary.
  - Resampled to 1 km grid for alignment.
  - Split: 80% train, 10% validation, 10% test.

## Model Architecture
- **Coordinated Encoders**:
  - LST/PM₂.₅: Three Conv–BN–ReLU stages with stride-2 downsampling, Kaiming initialization.
    
    <img width="1873" height="711" alt="image" src="https://github.com/user-attachments/assets/2e035634-418b-44dc-a5fc-00ed50884fb8" />

  - Land Cover: Embedding layer (37 classes → 8-dim), CoordConv, residual dilated block, CBAM attention.
    
    <img width="882" height="1119" alt="image" src="https://github.com/user-attachments/assets/efda79a7-74d4-4d37-8bae-c93a720ddb02" />

- **Attention U-Net as Feature Fusion Mechanism**:
  - Concatenates encoder outputs, uses attention gates to reweight spatial features, and outputs full-resolution UHII via 1×1 convolution.

    <img width="1592" height="911" alt="image" src="https://github.com/user-attachments/assets/273ceac4-0c4d-49f2-a824-e69108c96cae" />

- **Training Details**: MSE loss, Adam optimizer, CosineAnnealingLR scheduler, trained over 2003–2020 data.

## Results
- **Quantitative**:
  - Test MSE: 0.1104 (vs. baseline UNet: 0.185).
- **Qualitative**:
  - Attention maps highlight urban districts with high impervious surfaces and PM₂.₅ hotspots, aligning with known UHI patterns.
- **Interpretability**:
  - CBAM attention emphasizes pollution-heat synergies, with blue streaks in attention masks indicating high-influence regions (e.g., urban centers).

## Humanitarian Implications
- Identifies UHI drivers (impervious surfaces, PM₂.₅) to guide urban greening, zoning, and emission policies.
- Supports equitable interventions in Pune, where 2024 data show heat stress in low-income areas, potentially reducing ~15% of heat-related vulnerabilities.

## Citation
If you use this work, please cite:
```
Patil, A., Kalani, P., Lolage, O., Lanjewar, N. (2025). Learning Urban Heat Signatures via Coordinated Multi-Stream Convolution and Attention-Based Fusion. Vishwakarma Institute of Technology.
```
