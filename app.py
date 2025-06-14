from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import time
import threading
import json

app = Flask(__name__)
CORS(app)  # Enable cross-origin requests

# Global variables for training status
training_status = {
    'is_training': False,
    'current_iteration': 0,
    'current_accuracy': 0,
    'training_log': [],
    'final_results': None
}

# Load your data once at startup
try:
    data = pd.read_csv("mnist.csv")
    data = np.array(data)
    m, n = data.shape
    np.random.shuffle(data)
    
    # Split the data
    data_dev = data[0:1000].T
    Y_dev = data_dev[0]
    X_dev = data_dev[1:n] / 255.0  # Normalize
    
    data_train = data[1000:m].T
    Y_train = data_train[0]
    X_train = data_train[1:n] / 255.0  # Normalize
    
    print("Data loaded successfully!")
    print(f"Training samples: {X_train.shape[1]}")
    print(f"Validation samples: {X_dev.shape[1]}")
    
except Exception as e:
    print(f"Error loading data: {e}")
    X_train = Y_train = X_dev = Y_dev = None

# Neural Network Functions
def init_params(hidden1_size, hidden2_size=0):
    """Initialize network parameters based on architecture"""
    if hidden2_size > 0:
        # 3-layer network
        W1 = np.random.randn(hidden1_size, 784) * np.sqrt(2/784)
        b1 = np.zeros((hidden1_size, 1))
        W2 = np.random.randn(hidden2_size, hidden1_size) * np.sqrt(2/hidden1_size)
        b2 = np.zeros((hidden2_size, 1))
        W3 = np.random.randn(10, hidden2_size) * np.sqrt(2/hidden2_size)
        b3 = np.zeros((10, 1))
        return W1, b1, W2, b2, W3, b3
    else:
        # 2-layer network
        W1 = np.random.randn(hidden1_size, 784) * np.sqrt(2/784)
        b1 = np.zeros((hidden1_size, 1))
        W2 = np.random.randn(10, hidden1_size) * np.sqrt(2/hidden1_size)
        b2 = np.zeros((10, 1))
        return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    A = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return A / np.sum(A, axis=0, keepdims=True)

def forward_prop(W1, b1, W2, b2, X, W3=None, b3=None):
    """Forward propagation for 2 or 3 layer network"""
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    
    if W3 is not None:  # 3-layer network
        Z2 = W2.dot(A1) + b2
        A2 = ReLU(Z2)
        Z3 = W3.dot(A2) + b3
        A3 = softmax(Z3)
        return Z1, A1, Z2, A2, Z3, A3
    else:  # 2-layer network
        Z2 = W2.dot(A1) + b2
        A2 = softmax(Z2)
        return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y, W3=None, b3=None, Z3=None, A3=None):
    """Backward propagation for 2 or 3 layer network"""
    m = Y.size
    one_hot_Y = one_hot(Y)
    
    if W3 is not None:  # 3-layer network
        dZ3 = A3 - one_hot_Y
        dW3 = 1 / m * dZ3.dot(A2.T)
        db3 = 1 / m * np.sum(dZ3, axis=1, keepdims=True)
        dZ2 = W3.T.dot(dZ3) * (Z2 > 0)
        dW2 = 1 / m * dZ2.dot(A1.T)
        db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = W2.T.dot(dZ2) * (Z1 > 0)
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
        return dW1, db1, dW2, db2, dW3, db3
    else:  # 2-layer network
        dZ2 = A2 - one_hot_Y
        dW2 = 1 / m * dZ2.dot(A1.T)
        db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = W2.T.dot(dZ2) * (Z1 > 0)
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
        return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha, W3=None, b3=None, dW3=None, db3=None):
    """Update parameters for 2 or 3 layer network"""
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    
    if W3 is not None:
        W3 = W3 - alpha * dW3
        b3 = b3 - alpha * db3
        return W1, b1, W2, b2, W3, b3
    else:
        return W1, b1, W2, b2

def get_predictions(A):
    return np.argmax(A, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def analyze_training_results(final_accuracy, val_accuracy, training_time, iterations, params):
    """Analyze training results and provide specific recommendations"""
    recommendations = []
    priority_level = "low"  # low, medium, high
    
    # Extract key parameters
    learning_rate = params['learningRate']
    hidden1_size = params['hidden1Size']
    hidden2_size = params['hidden2Size']
    lr_decay = params['lrDecay']
    
    # Calculate overfitting gap
    overfitting_gap = final_accuracy - val_accuracy
    
    # Performance Analysis
    if final_accuracy < 0.7:
        priority_level = "high"
        recommendations.append({
            'type': 'critical',
            'title': 'Low Training Accuracy',
            'issue': f'Training accuracy is only {final_accuracy*100:.1f}%',
            'recommendation': 'Increase hidden layer size to 256+ neurons or add a second hidden layer',
            'specific_action': 'Set Hidden Layer 1 to 256 and Hidden Layer 2 to 128'
        })
        
        if learning_rate < 0.1:
            recommendations.append({
                'type': 'critical',
                'title': 'Learning Rate Too Low',
                'issue': f'Learning rate {learning_rate} is too conservative',
                'recommendation': 'Increase learning rate for faster convergence',
                'specific_action': 'Try learning rate 0.2 or 0.3'
            })
    
    elif final_accuracy < 0.85:
        priority_level = "medium"
        recommendations.append({
            'type': 'improvement',
            'title': 'Moderate Performance',
            'issue': f'Training accuracy {final_accuracy*100:.1f}% has room for improvement',
            'recommendation': 'Consider increasing network capacity or training longer',
            'specific_action': f'Increase hidden layer size to {hidden1_size + 64} or train for {iterations + 200} iterations'
        })
    
    # Overfitting Analysis
    if overfitting_gap > 0.1:
        priority_level = max(priority_level, "medium")
        recommendations.append({
            'type': 'warning',
            'title': 'Overfitting Detected',
            'issue': f'Gap between training ({final_accuracy*100:.1f}%) and validation ({val_accuracy*100:.1f}%) is {overfitting_gap*100:.1f}%',
            'recommendation': 'Reduce model complexity or add regularization',
            'specific_action': 'Decrease hidden layer sizes by 25% or increase dropout rate to 0.3-0.5'
        })
    
    # Underfitting Analysis
    elif overfitting_gap < 0.02 and final_accuracy < 0.9:
        recommendations.append({
            'type': 'improvement',
            'title': 'Possible Underfitting',
            'issue': 'Model may be too simple for the data complexity',
            'recommendation': 'Increase model capacity',
            'specific_action': f'Increase Hidden Layer 1 to {min(hidden1_size * 2, 512)} neurons'
        })
    
    # Learning Rate Analysis
    convergence_rate = final_accuracy / (iterations / 100)  # Rough convergence speed
    
    if convergence_rate < 0.005 and learning_rate > 0.3:
        recommendations.append({
            'type': 'warning',
            'title': 'Learning Rate Too High',
            'issue': 'Training appears unstable, possibly overshooting optima',
            'recommendation': 'Reduce learning rate for more stable convergence',
            'specific_action': f'Try learning rate {learning_rate * 0.5:.3f}'
        })
    
    elif convergence_rate < 0.003 and learning_rate < 0.15:
        recommendations.append({
            'type': 'improvement',
            'title': 'Slow Convergence',
            'issue': 'Training is converging very slowly',
            'recommendation': 'Consider increasing learning rate or using learning rate scheduling',
            'specific_action': f'Try learning rate {min(learning_rate * 1.5, 0.4):.3f}'
        })
    
    # Architecture Analysis
    if hidden2_size == 0 and final_accuracy < 0.9:
        recommendations.append({
            'type': 'improvement',
            'title': 'Single Hidden Layer Limitation',
            'issue': 'Single layer may limit learning complex patterns',
            'recommendation': 'Add a second hidden layer for deeper feature learning',
            'specific_action': f'Set Hidden Layer 2 to {hidden1_size // 2} neurons'
        })
    
    # Training Duration Analysis
    iterations_per_percent = iterations / (final_accuracy * 100)
    if iterations_per_percent > 15 and final_accuracy < 0.9:
        recommendations.append({
            'type': 'improvement',
            'title': 'Inefficient Training',
            'issue': 'Requiring too many iterations for accuracy gained',
            'recommendation': 'Optimize learning rate or try better initialization',
            'specific_action': 'Increase learning rate by 50% and reduce iterations by 25%'
        })
    
    # Learning Rate Decay Analysis
    if lr_decay < 0.95 and final_accuracy > 0.85:
        recommendations.append({
            'type': 'optimization',
            'title': 'Aggressive Learning Rate Decay',
            'issue': 'Learning rate decay might be too aggressive for fine-tuning',
            'recommendation': 'Use gentler decay for better final performance',
            'specific_action': 'Set learning rate decay to 0.98 or 0.99'
        })
    
    # Exceptional Performance
    if final_accuracy > 0.92 and abs(overfitting_gap) < 0.05:
        recommendations.append({
            'type': 'success',
            'title': 'Excellent Performance!',
            'issue': 'Model is performing very well with good generalization',
            'recommendation': 'Consider experimenting with even more complex architectures',
            'specific_action': 'Try 3-layer network: 512->256->128 or experiment with CNN layers'
        })
    
    # No recommendations case
    if not recommendations:
        recommendations.append({
            'type': 'success',
            'title': 'Good Configuration',
            'issue': 'Current parameters are working reasonably well',
            'recommendation': 'Fine-tune existing parameters or try small incremental changes',
            'specific_action': 'Experiment with ±20% changes to learning rate and hidden layer sizes'
        })
    
    return {
        'recommendations': recommendations,
        'priority_level': priority_level,
        'summary': {
            'performance_tier': 'excellent' if final_accuracy > 0.9 else 'good' if final_accuracy > 0.8 else 'needs_improvement',
            'overfitting_status': 'overfitting' if overfitting_gap > 0.1 else 'underfitting' if overfitting_gap < 0.02 else 'balanced',
            'convergence_speed': 'fast' if convergence_rate > 0.008 else 'slow' if convergence_rate < 0.004 else 'normal'
        }
    }

def train_network(params):
    """Train the neural network with given parameters"""
    global training_status
    
    # Reset training status
    training_status = {
        'is_training': True,
        'current_iteration': 0,
        'current_accuracy': 0,
        'training_log': [],
        'final_results': None
    }
    
    try:
        # Extract parameters
        learning_rate = params['learningRate']
        iterations = params['iterations']
        hidden1_size = params['hidden1Size']
        hidden2_size = params['hidden2Size']
        lr_decay = params['lrDecay']
        
        # Initialize network
        if hidden2_size > 0:
            W1, b1, W2, b2, W3, b3 = init_params(hidden1_size, hidden2_size)
        else:
            W1, b1, W2, b2 = init_params(hidden1_size)
            W3 = b3 = None
        
        training_status['training_log'].append("Neural network initialized")
        training_status['training_log'].append(f"Architecture: 784 -> {hidden1_size} -> {hidden2_size if hidden2_size > 0 else ''} -> 10")
        training_status['training_log'].append(f"Learning rate: {learning_rate}")
        training_status['training_log'].append("Starting training...")
        
        start_time = time.time()
        
        # Training loop
        for i in range(iterations):
            current_lr = learning_rate * (lr_decay ** (i // 50))
            
            # Forward propagation
            if W3 is not None:
                Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, X_train, W3, b3)
                predictions = get_predictions(A3)
                accuracy = get_accuracy(predictions, Y_train)
                
                # Backward propagation
                dW1, db1, dW2, db2, dW3, db3 = backward_prop(Z1, A1, Z2, A2, W1, W2, X_train, Y_train, W3, b3, Z3, A3)
                
                # Update parameters
                W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, current_lr, W3, b3, dW3, db3)
            else:
                Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X_train)
                predictions = get_predictions(A2)
                accuracy = get_accuracy(predictions, Y_train)
                
                # Backward propagation
                dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X_train, Y_train)
                
                # Update parameters
                W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, current_lr)
            
            # Update status
            training_status['current_iteration'] = i
            training_status['current_accuracy'] = accuracy
            
            # Log progress
            if i % 25 == 0:
                log_msg = f"Iteration {i}: Accuracy {accuracy:.4f} ({accuracy*100:.2f}%) - LR: {current_lr:.4f}"
                training_status['training_log'].append(log_msg)
                print(log_msg)
        
        # Calculate final results
        training_time = time.time() - start_time
        
        # Validation accuracy
        if W3 is not None:
            _, _, _, _, _, A3_val = forward_prop(W1, b1, W2, b2, X_dev, W3, b3)
            val_predictions = get_predictions(A3_val)
        else:
            _, _, _, A2_val = forward_prop(W1, b1, W2, b2, X_dev)
            val_predictions = get_predictions(A2_val)
        
        val_accuracy = get_accuracy(val_predictions, Y_dev)
        
        training_status['final_results'] = {
            'final_accuracy': accuracy,
            'validation_accuracy': val_accuracy,
            'training_time': training_time,
            'total_iterations': iterations
        }
        
        # Generate intelligent recommendations
        analysis = analyze_training_results(accuracy, val_accuracy, training_time, iterations, params)
        training_status['analysis'] = analysis
        
        training_status['training_log'].append("=" * 50)
        training_status['training_log'].append("TRAINING COMPLETED")
        training_status['training_log'].append(f"Final Training Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        training_status['training_log'].append(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        training_status['training_log'].append(f"Training Time: {training_time:.1f} seconds")
        
        # Add key insights to log
        training_status['training_log'].append("")
        training_status['training_log'].append("PERFORMANCE ANALYSIS:")
        overfitting_gap = accuracy - val_accuracy
        if overfitting_gap > 0.1:
            training_status['training_log'].append(f"WARNING: Overfitting detected: {overfitting_gap*100:.1f}% gap")
        elif overfitting_gap < 0.02:
            training_status['training_log'].append("INFO: Possible underfitting: very small train/val gap")
        else:
            training_status['training_log'].append("SUCCESS: Good generalization: balanced train/val performance")
            
        training_status['training_log'].append(f"Performance tier: {analysis['summary']['performance_tier']}")
        training_status['training_log'].append(f"Top recommendation: {analysis['recommendations'][0]['title']}")
        
    except Exception as e:
        training_status['training_log'].append(f"ERROR: {str(e)}")
        print(f"Training error: {e}")
    
    training_status['is_training'] = False

@app.route('/train', methods=['POST'])
def start_training():
    """Start training with given parameters"""
    if X_train is None:
        return jsonify({'error': 'Dataset not loaded'}), 400
    
    if training_status['is_training']:
        return jsonify({'error': 'Training already in progress'}), 400
    
    params = request.json
    
    # Start training in background thread
    training_thread = threading.Thread(target=train_network, args=(params,))
    training_thread.start()
    
    return jsonify({'message': 'Training started', 'status': 'success'})

@app.route('/status', methods=['GET'])
def get_status():
    """Get current training status"""
    return jsonify(training_status)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'data_loaded': X_train is not None,
        'training_samples': X_train.shape[1] if X_train is not None else 0
    })

if __name__ == '__main__':
    print("Starting Flask server...")
    print("Make sure mnist.csv is in the same directory!")
    
    # Environment configuration for security and flexibility
    import os
    
    # Default to secure production settings
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    host = os.environ.get('FLASK_HOST', '127.0.0.1')  # Default to localhost only
    port = int(os.environ.get('FLASK_PORT', '5000'))
    
    if debug_mode:
        print("⚠️  Running in DEBUG mode - only use for development!")
    else:
        print("✅ Running in production mode")
        
    if host == '0.0.0.0':
        print("⚠️  Server accessible from network - ensure this is intentional!")
    else:
        print(f"🔒 Server bound to {host} (localhost only)")
    
    print(f"🚀 Starting server on http://{host}:{port}")
    app.run(debug=debug_mode, host=host, port=port)