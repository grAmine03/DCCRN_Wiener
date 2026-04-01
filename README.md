## DCCRN: Deep Complex Convolution Recurrent Network for Phase-Aware Speech Enhancement
__Authors__: Mohamed Amine GRINI, Yanis ALLAL

Although deep learning has significantly advanced speech enhance-
ment, traditional time-frequency (TF) domain methods often focus
solely on magnitude, treating phase as intractable. This ignores
critical alignment information required for high-fidelity audio re-
construction. Recent continuous-time models attempt complex-
spectral mapping but often rely on real-valued networks. In this
review, we examine the Deep Complex Convolution Recurrent
Network (DCCRN) [ 1 ], a framework that integrates a Convolu-
tional Encoder-Decoder (CED) structure with Long Short-Term
Memory (LSTM) entirely within the complex domain. We imple-
ment and compare three methods: a sliding-window Wiener filter
baseline, a magnitude-only convolutional neural network (CRN),
and a complex-valued CNN (DCUNET) that explicitly processes
real and imaginary STFT components using complex convolutions.
Experimental evaluation on speech corrupted by stationary noise
at various SNR levels (0–20 dB) shows that complex-valued net-
works achieve 2–3 dB higher SI-SNR and substantially better phase
preservation compared to magnitude-only baselines, empirically
validating DCCRN’s core hypothesis that explicit phase modeling
improves speech enhancement quality


Original Paper: https://arxiv.org/abs/2008.00264

Sample: https://huyanxin.github.io/DeepComplexCRN/

