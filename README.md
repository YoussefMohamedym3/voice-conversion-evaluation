# Voice Conversion Quality Metrics

When converting a voice, it is crucial to:

1. Preserve the **pitch** of the source voice in the converted voice.
2. Apply the **timbre** of the target voice in the converted voice.
3. Ensure no **noise** is added to the converted voice during the process.

## Key Definitions

- **Source Voice**: The original voice whose pitch is preserved during the conversion process.
- **Target Voice**: The voice whose timbre is applied in the converted voice.
- **Converted Voice**: The output of the voice conversion system, combining the pitch of the source voice and the timbre of the target voice.

## Evaluation Metrics and Code References

### 1. **Mel-Cepstral Distortion (MCD)**

- **Code Reference**: `mcd.py`
- **Functionality**:
  - Computes MCD between the Mel-cepstral coefficients (MCEPs) of target and converted audio.
  - Extracts MCEPs using the WORLD vocoder and `pysptk` library.
  - Calculates MCD in decibels (dB), indicating spectral distance.
- **Importance**:
  - **Spectral Accuracy**: Measures how close the converted audio's spectral envelope is to the target.
  - Lower MCD values (4–6 dB for high-quality conversion) represent better spectral similarity.
- **Unit**:
  - Lower MCD values (in dB) indicate better conversion performance. Typically, values between 4 and 6 dB represent high-quality conversions.

### 2. **Fundamental Frequency RMSE (F0_RMSE) & logF0 RMSE**

- **Code Reference**: `F0_RMSE_&_logF0RMSE.py`
- **Functionality**:
  - Calculates RMSE for F0 (pitch) between source and converted audio.
  - Computes RMSE of logarithmic F0 values for finer pitch variation analysis.
- **Importance**:
  - **Pitch Preservation**: Evaluates similarity of pitch between source and converted audio.
  - Lower RMSE (0.1–0.5 for good performance) reflects better pitch alignment.
- **Unit**:
  - Lower RMSE values indicate better alignment of pitch between audio signals. Typical Range: 0.1 to 0.5 for good performance.

### 3. **Mean Squared Deviation (MSD)**

- **Code Reference**: `MSD.py`
- **Functionality**:
  - Analyzes temporal dynamics by computing MSD between modulation spectra of target and converted audio.
- **Importance**:
  - **Temporal Similarity**: Assesses how well the temporal dynamics of converted audio match the target.
- **Unit**:
  - Lower MSD values indicate better preservation of temporal dynamics.

### 4. **Global Variance (GV)**

- **Code Reference**: `GV.py`
- **Functionality**:
  - Computes variance of static features (e.g., MCEPs) for target and converted audio.
- **Importance**:
  - **Expressiveness**: Evaluates whether converted audio retains the natural variance needed for naturalness.
- **Unit**:
  - Higher similarity in GV between target and converted audio indicates better performance.

### 5. **Signal-to-Noise Ratio (SNR)**

- **Code Reference**: `SNR.py`
- **Functionality**:
  - Measures the ratio of signal power to noise power in decibels (dB).
- **Importance**:
  - **Quality Assessment**: Higher SNR reflects less noise and better signal preservation.
- **Unit**:
  - Decibels (dB).
  - Higher SNR values indicate less noise and better signal quality.
  - Lower SNR values indicate more noise or distortion.

### 6. **Mel-Cepstral Coefficient (MCEP) Trajectories**

- **Code Reference**: `MCEP_Trajectory.py`
- **Functionality**:
  - Visualizes MCEP trajectories for selected dimensions of target and converted audio.
- **Importance**:
  - **Spectral Dynamics**: Reflects spectral envelope dynamics and temporal coherence.
- **Unit**:
  - MCEP values plotted against frame indices.
  - Closer MCEP trajectories between target and converted audio indicate better voice conversion quality and spectral similarity.

### 7. **Mean Mel-Cepstral Coefficients (MCEP)**

- **Code Reference**: `Mean_MCEP.py`
- **Functionality**:
  - Visualizes mean MCEP values for target and converted audio.
- **Importance**:
  - **Spectral Similarity**: Summarizes average spectral characteristics.
- **Unit**:
  - Mean MCEP values plotted against MCEP dimensions.
  - Closer mean MCEP distributions between target and converted audio indicate better preservation of spectral features.

### 8. **Scatter Plots of MCEP**

- **Code Reference**: `MCEP_Scatter_Plot.py`
- **Functionality**:
  - Generates scatter plots to compare MCEP dimensions for target and converted audio.
- **Importance**:
  - **Dimensional Analysis**: Highlights correlations in spectral features.
- **Unit**:
  - Closer and more aligned distributions between target and converted audio indicate better spectral feature matching.

### 9. **Modulation Spectrum**

- **Code Reference**: `Modulation_Spectrum.py`
- **Functionality**:
  - Analyzes temporal and spectral dynamics through FFT-based modulation spectrum.
- **Importance**:
  - **Temporal-Spectral Analysis**: Measures preservation of modulations.
- **Unit**:
  - Log-scaled modulation spectrum plots for selected dimensions.
  - Closer and more aligned spectra between target and converted audio indicate better temporal and spectral feature preservation.

### 10. **MOS Prediction**

- **Code Reference**: `mos.py`
- **Functionality**:
  - Predicts Mean Opinion Score (MOS) using MBNet for perceptual quality evaluation.
- **Importance**:
  - **Quality Assessment**: Automates subjective quality benchmarking.
- **Unit**:
  - MOS (Mean Opinion Score): A scalar valueon a scale from 1 (Bad) to 5 (Excellent), where higher MOS indicates better subjective quality.

### 11. **Speaker Verification**

- **Code Reference**: `Speaker_Verification.py`
- **Functionality**:
  - Calculates speaker verification acceptance rate (SVAR) using a pre-trained VoiceEncoder model.
- **Importance**:
  - **Speaker Identity**: Evaluates how well the speaker identity is preserved.
  - **Acceptance Rate**: Ranges from 0 (no matches) to 1 (perfect matches).
- **Unit**:
  - Speaker Verification Acceptance Rate (SVAR): A ratio ranging from 0 (no matches) to 1 (perfect matches), representing the proportion of audio pairs that meet or exceed the threshold for speaker similarity.

---

## Recreating the Environment

To recreate the environment, follow these steps:

1. Use the following command to create the environment:

```bash
conda env create -f environment.yml
```

2. Once the environment is created, activate it:

```bash
conda activate myenv
```

3. You can verify that the environment is set up correctly by checking the installed packages:

```bash
conda list
```
