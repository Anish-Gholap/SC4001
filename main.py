"""
Main script to run the emotion classification project.
"""
import argparse
import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """
    Main entry point for the emotion classification project.
    Allows selection of different modes: train, test, predict, or compare.
    """
    parser = argparse.ArgumentParser(description="Emotion Classification with PyTorch")
    parser.add_argument(
        '--mode', 
        type=str, 
        default='train_lstm',
        choices=['train_lstm', 'train_rnn', 'test', 'predict', 'compare'],
        help='Mode to run the project in'
    )
    parser.add_argument(
        '--text', 
        type=str, 
        help='Text to predict emotion for (only used in predict mode)'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default='lstm',
        choices=['lstm', 'rnn'],
        help='Model to use for testing or prediction'
    )
    args = parser.parse_args()

    # Based on the mode, run the appropriate script
    try:
        if args.mode == 'train_lstm':
            print("Training LSTM model...")
            # Use dynamic import to locate the module
            import importlib
            train_lstm_module = importlib.import_module('train_lstm')
            train_lstm_module.main()
        
        elif args.mode == 'train_rnn':
            print("Training simple RNN model...")
            import importlib
            train_rnn_module = importlib.import_module('train_simple_rnn')
            train_rnn_module.main()
        
        elif args.mode == 'test':
            print(f"Testing {args.model} model...")
            import importlib
            if args.model == 'lstm':
                test_module = importlib.import_module('train_lstm')
                test_module.main()
            else:
                test_module = importlib.import_module('train_simple_rnn')
                test_module.main()
        
        elif args.mode == 'predict':
            print(f"Predicting emotion using {args.model} model...")
            import importlib
            inference_module = importlib.import_module('inference')
            
            model_path = 'best_emotion_classifier.pth' if args.model == 'lstm' else 'best_simple_rnn_classifier.pth'
            resources = inference_module.load_inference_resources(model_path, 'vocab.pkl', 'label_encoder.pkl')
            
            # Get text from command line or prompt
            text = args.text
            if not text:
                text = input("Enter text to analyze: ")
            
            # Predict emotion
            result = inference_module.predict_emotion(text, resources)
            
            # Print results
            print(f"\nText: {result['text']}")
            print(f"Predicted emotion: {result['predicted_emotion']} (Confidence: {result['confidence']:.2f}%)")
            print("\nAll emotions:")
            for emotion, prob in result['all_emotions']:
                print(f"  {emotion}: {prob:.2f}%")
        
        elif args.mode == 'compare':
            print("Comparing models...")
            import importlib
            compare_module = importlib.import_module('compare_models')
            compare_module.main()
    
    except ModuleNotFoundError as e:
        print(f"Error: Could not find module. {e}")
        print("\nPossible causes:")
        print("1. Make sure all Python files are in the correct directory")
        print("2. Check that file names match the imports")
        print("3. Make sure you're running from the project root directory")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
    
    else:
        print(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()