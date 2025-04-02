"""
Main script to run the emotion classification project.
"""
import argparse
import os
import sys

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
    if args.mode == 'train_lstm':
        print("Training LSTM model...")
        from train_lstm import main as train_lstm_main
        train_lstm_main()
    
    elif args.mode == 'train_rnn':
        print("Training simple RNN model...")
        from train_simple_rnn import main as train_rnn_main
        train_rnn_main()
    
    elif args.mode == 'test':
        print(f"Testing {args.model} model...")
        if args.model == 'lstm':
            from train_lstm import main as test_lstm_main
            test_lstm_main()
        else:
            from train_simple_rnn import main as test_rnn_main
            test_rnn_main()
    
    elif args.mode == 'predict':
        print(f"Predicting emotion using {args.model} model...")
        from inference import load_inference_resources, predict_emotion
        
        model_path = 'best_emotion_classifier.pth' if args.model == 'lstm' else 'best_simple_rnn_classifier.pth'
        resources = load_inference_resources(model_path, 'vocab.pkl', 'label_encoder.pkl')
        
        # Get text from command line or prompt
        text = args.text
        if not text:
            text = input("Enter text to analyze: ")
        
        # Predict emotion
        result = predict_emotion(text, resources)
        
        # Print results
        print(f"\nText: {result['text']}")
        print(f"Predicted emotion: {result['predicted_emotion']} (Confidence: {result['confidence']:.2f}%)")
        print("\nAll emotions:")
        for emotion, prob in result['all_emotions']:
            print(f"  {emotion}: {prob:.2f}%")
    
    elif args.mode == 'compare':
        print("Comparing models...")
        from compare_models import main as compare_main
        compare_main()
    
    else:
        print(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()