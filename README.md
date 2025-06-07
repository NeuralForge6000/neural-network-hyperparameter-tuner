# Neural Network Hyperparameter Tuner

An interactive web application for training and optimizing neural networks on the Fashion-MNIST dataset. Features real-time training with intelligent AI-powered recommendations for hyperparameter tuning.

![Neural Network Tuner](https://img.shields.io/badge/Neural%20Network-Tuner-brightgreen)
![Python](https://img.shields.io/badge/Python-3.13-blue)
![Flask](https://img.shields.io/badge/Flask-2.3.3-red)

## Features

- **Interactive Web Interface**: Matrix-themed UI with real-time parameter sliders
- **Live Training**: Watch your neural network train in real-time
- **AI-Powered Recommendations**: Get intelligent suggestions for improving performance
- **Quick Apply**: One-click application of recommended parameter changes
- **Performance Analysis**: Detailed analysis of overfitting, underfitting, and convergence
- **Custom Architecture**: Support for 2-3 layer neural networks with configurable sizes

## Architecture

- **Backend**: Flask server running the neural network training
- **Frontend**: HTML/CSS/JavaScript matrix-themed interface
- **Neural Network**: From-scratch implementation using NumPy
- **Dataset**: Fashion-MNIST (10 clothing categories)

## Installation

### Prerequisites
- Python 3.13+
- Fashion-MNIST dataset (`mnist.csv`)

### Clone and Install
1. Clone the repository:

1. **Create Kaggle Account** (if you don't have one):
   - Go to [kaggle.com](https://www.kaggle.com) and sign up

2. **Download Fashion-MNIST**:
   - Visit [Fashion-MNIST Dataset](https://www.kaggle.com/datasets/zalando-research/fashionmnist/data)
   - Click "Download" button
   - Extract the downloaded ZIP file

3. **Prepare the Data**:
   - Look for `fashion-mnist_train.csv` or similar file
   - Rename it to `mnist.csv`
   - Copy to your project directory:
   ```
   neural-network-tuner/
   ├── mnist.csv  ← Place the dataset file here
   ├── app.py
   ├── neural_network_tuner.html
   └── ...
   ```

### Alternative: Manual Download
If you have issues with Kaggle, you can also find Fashion-MNIST datasets on:
- [GitHub zalando/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)
- Various ML dataset repositories

Just ensure the final CSV has the same format (first column = labels, remaining 784 columns = pixel values).
```bash
git clone https://github.com/yourusername/neural-network-tuner.git
cd neural-network-tuner
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the Fashion-MNIST dataset:
   - Go to [Kaggle Fashion-MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist/data)
   - Download the dataset (you may need to create a free Kaggle account)
   - Extract and rename the CSV file to `mnist.csv`
   - Place it in the project root directory

## Usage

### 1. Start the Flask Server
```bash
python app.py
```

You should see:
```
Starting Flask server...
Data loaded successfully!
Training samples: 41000
Validation samples: 1000
 * Running on http://127.0.0.1:5000
```

### 2. Open the Web Interface
Open `neural_network_tuner.html` in your web browser.

### 3. Configure Parameters
Adjust the hyperparameters using the sliders:

**Training Parameters:**
- **Learning Rate** (0.01-1.0): Controls optimization step size
- **Iterations** (100-2000): Number of training cycles
- **Batch Size** (16-256): Samples per weight update

**Network Architecture:**
- **Hidden Layer 1** (32-512): Primary feature extraction layer
- **Hidden Layer 2** (0-256): Secondary feature layer (0 = disabled)
- **Dropout Rate** (0-0.8): Regularization strength

**Optimization:**
- **Learning Rate Decay** (0.9-1.0): Gradual LR reduction
- **Weight Decay** (0-0.01): L2 regularization
- **Momentum** (0-0.99): Gradient momentum

### 4. Train and Optimize
1. Click "INITIALIZE TRAINING SEQUENCE"
2. Watch real-time training progress
3. Review AI-generated recommendations
4. Use "Quick Apply" to implement suggestions
5. Retrain with optimized parameters

## Performance Expectations

- **Baseline**: ~75-85% accuracy with default settings
- **Optimized**: 90-95% accuracy with proper tuning
- **Training Time**: 30-120 seconds depending on parameters

## AI Recommendations System

The system analyzes your training results and provides specific recommendations:

### Critical Issues
- Low accuracy detection and solutions
- Learning rate optimization
- Architecture improvements

### Performance Optimization
- Overfitting/underfitting detection
- Convergence speed analysis
- Parameter efficiency suggestions

### Success Recognition
- Celebrates excellent performance
- Suggests advanced experiments

## Project Structure

```
neural-network-tuner/
├── app.py                      # Flask backend server
├── neural_network_tuner.html   # Web interface
├── MNIST_Project.ipynb         # Original Jupyter notebook
├── requirements.txt            # Python dependencies
├── .gitignore                 # Git ignore rules
├── README.md                  # This file
└── mnist.csv                  # Fashion-MNIST dataset (not included)
```

## API Endpoints

- `POST /train` - Start training with parameters
- `GET /status` - Get current training status
- `GET /health` - Health check

## Technical Details

### Neural Network Implementation
- **Framework**: Pure NumPy implementation
- **Architecture**: Fully connected layers with ReLU activation
- **Optimization**: Gradient descent with momentum and decay
- **Regularization**: Dropout and L2 weight decay

### Web Technologies
- **Backend**: Flask with CORS support
- **Frontend**: Vanilla JavaScript with CSS animations
- **Styling**: Matrix-themed dark UI with green accents
- **Fonts**: Orbitron and Source Code Pro

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Future Enhancements

- [ ] Convolutional Neural Network support
- [ ] Transfer learning capabilities
- [ ] Model export/import functionality
- [ ] Advanced optimizers (Adam, RMSprop)
- [ ] Automatic hyperparameter search
- [ ] Mobile-responsive design

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Fashion-MNIST dataset by Zalando Research
- Neural network concepts from CS231n Stanford
- Matrix visual effects inspired by The Matrix (1999)

## Dataset Information

**Fashion-MNIST** consists of 70,000 grayscale images in 10 categories:
- 0: T-shirt/top
- 1: Trouser  
- 2: Pullover
- 3: Dress
- 4: Coat
- 5: Sandal
- 6: Shirt
- 7: Sneaker
- 8: Bag
- 9: Ankle boot

Each image is 28x28 pixels, flattened to 784 features for the neural network.

**Dataset Source:** [Zalando Research Fashion-MNIST on Kaggle](https://www.kaggle.com/datasets/zalando-research/fashionmnist/data)

**Note:** The dataset file should be renamed to `mnist.csv` and placed in the project root directory. This file is not included in the repository due to its size (~109MB).