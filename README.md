# Y-Net

Code of the main function of the network model：model/Y_MBNet.py.

Dataset  URL: https://ieee-dataport.org/documents/datasets-based-x-spacemethod-magnetic-particle-imaging.

if __name__ == '__main__':
    # 创建模型
    model = Y_MBNet1(in_channels=1, him_channels=2, out_channels=1)
    # 创建模拟数据
    input_data = torch.randn(10, 1, 51, 51)
    # 模型前向传播
    output_data = model(input_data)
    print("Output1 Shape:", output_data.shape)

The network is implemented in PyTorch and trained on a single NVIDIA GeForce RTX 2080 Ti GPU for 2000 epochs with a batch size of 64. The optimizer is Adam with an initial learning rate of 1 × 10^{-4}. 

The Spatial Mamba Branch (SM) and Channel Mamba Branch (CM) are configured with channel sizes of 16, 32, and 64. The Reconstruction Module includes two stages Dense Blocks (DBs) (five convolutional layers per block) and transposed convolutions (ConvTranspose, 3 × 3, stride=2) with output channels of 64 and 32, respectively. The final refinement layers consist of a 3 × 3 convolution with 16 channels and a 1 × 1 convolution with 1 channel, producing the reconstructed output of size 1 ×s H × W.
