"""
Enhanced Explainability Analysis with English Labels and PDF Report Generation
Based on multi-channel bearing fault diagnosis using SCTL-FD framework
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

# Set English fonts
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models import create_lstm_classification_model, create_lstm_supcon_model, load_pretrained_encoder
from src.datasets import StandardDataset, SupConDataset

class EnglishExplainabilityAnalyzer:
    """Enhanced explainability analyzer with English labels and PDF generation"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Fault type mapping (English)
        self.fault_names = ['Normal', 'Inner Fault', 'Outer Fault', 'Ball Fault']
        self.fault_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Professional colors

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Initialize models
        self.pretrained_model = None
        self.final_model = None

    def load_models(self):
        """Load pretrained and final models"""
        print("Loading multi-channel trained models...")

        # Load pretrained encoder model
        self.pretrained_model = create_lstm_supcon_model(
            signal_length=self.args.signal_length,
            feature_dim=self.args.feature_dim,
            hidden_dim=self.args.hidden_dim,
            num_layers=self.args.num_layers
        ).to(self.device)

        if os.path.exists(self.args.pretrained_encoder_path):
            encoder_state = torch.load(self.args.pretrained_encoder_path, map_location=self.device)
            self.pretrained_model.encoder.load_state_dict(encoder_state)
            self.pretrained_model.eval()
            print("✓ Pretrained encoder loaded successfully")
        else:
            print("⚠ Pretrained encoder not found")

        # Load final transfer learning model
        self.final_model = create_lstm_classification_model(
            signal_length=self.args.signal_length,
            feature_dim=self.args.feature_dim,
            hidden_dim=self.args.hidden_dim,
            num_layers=self.args.num_layers,
            num_classes=self.args.num_classes
        ).to(self.device)

        if os.path.exists(self.args.final_model_path):
            model_state = torch.load(self.args.final_model_path, map_location=self.device)
            self.final_model.load_state_dict(model_state)
            self.final_model.eval()
            print("✓ Final model loaded successfully")
        else:
            print("⚠ Final model not found")

    def load_data(self):
        """Load multi-channel validation data"""
        print("Loading multi-channel data...")

        # Load validation data
        self.val_dataset = StandardDataset(self.args.val_data_path)
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=4
        )

        print(f"Validation data: {len(self.val_dataset)} samples")
        print(f"Label distribution: {np.bincount(self.val_dataset.labels)}")

    def extract_features(self, model, data_loader, model_type='pretrained'):
        """Extract features from model"""
        features_list = []
        labels_list = []
        predictions_list = []

        model.eval()
        with torch.no_grad():
            for data, labels in data_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)

                if model_type == 'pretrained':
                    # For pretrained SupCon model
                    feature = model.encoder(data)
                    predictions = None
                elif model_type == 'final':
                    # For final classification model
                    feature, logits = model(data)
                    predictions = torch.argmax(logits, dim=1)
                    predictions_list.extend(predictions.cpu().numpy())

                features_list.append(feature.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())

        features = np.concatenate(features_list, axis=0)
        labels = np.array(labels_list)
        predictions = np.array(predictions_list) if predictions_list else None

        return features, labels, predictions

    def analyze_feature_transfer(self):
        """Analyze feature transfer process with 3D visualization"""
        print("=== Transfer Process Explainability: Feature Transfer Analysis ===")

        # Extract pretrained features
        print("Extracting pretrained features (multi-channel)...")
        pretrained_features, pretrained_labels, _ = self.extract_features(
            self.pretrained_model, self.val_loader, 'pretrained'
        )

        # Extract final features
        print("Extracting fine-tuned features (multi-channel)...")
        final_features, final_labels, predictions = self.extract_features(
            self.final_model, self.val_loader, 'final'
        )

        # Perform 3D t-SNE dimensionality reduction
        print("Performing 3D t-SNE dimensionality reduction...")

        # Sample data for t-SNE (to avoid memory issues)
        n_samples = min(2000, len(pretrained_features))
        indices = np.random.choice(len(pretrained_features), n_samples, replace=False)

        # Combine features for joint t-SNE
        combined_features = np.vstack([
            pretrained_features[indices],
            final_features[indices]
        ])

        # 3D t-SNE
        tsne = TSNE(n_components=3, random_state=42, perplexity=30)
        features_3d = tsne.fit_transform(combined_features)

        # Split back
        pretrained_3d = features_3d[:n_samples]
        final_3d = features_3d[n_samples:]
        labels_sample = pretrained_labels[indices]

        # Create 3D visualization
        fig = plt.figure(figsize=(20, 8))

        # Pretrained features (left)
        ax1 = fig.add_subplot(121, projection='3d')
        for i, (fault_name, color) in enumerate(zip(self.fault_names, self.fault_colors)):
            mask = labels_sample == i
            if np.sum(mask) > 0:
                ax1.scatter(pretrained_3d[mask, 0], pretrained_3d[mask, 1], pretrained_3d[mask, 2],
                           c=color, label=fault_name, alpha=0.7, s=20)

        ax1.set_title('Pretrained Features (Multi-channel)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('t-SNE Component 1')
        ax1.set_ylabel('t-SNE Component 2')
        ax1.set_zlabel('t-SNE Component 3')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Fine-tuned features (right)
        ax2 = fig.add_subplot(122, projection='3d')
        for i, (fault_name, color) in enumerate(zip(self.fault_names, self.fault_colors)):
            mask = labels_sample == i
            if np.sum(mask) > 0:
                ax2.scatter(final_3d[mask, 0], final_3d[mask, 1], final_3d[mask, 2],
                           c=color, label=fault_name, alpha=0.7, s=20)

        ax2.set_title('Fine-tuned Features (Multi-channel)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('t-SNE Component 1')
        ax2.set_ylabel('t-SNE Component 2')
        ax2.set_zlabel('t-SNE Component 3')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle('3D Feature Space Transfer Analysis (Multi-channel)', fontsize=16, fontweight='bold')
        plt.tight_layout()

        return fig

    def analyze_inter_class_distances(self):
        """Analyze inter-class distances"""
        # Extract final features for analysis
        final_features, final_labels, _ = self.extract_features(
            self.final_model, self.val_loader, 'final'
        )

        # Calculate centroids for each class
        centroids = []
        for i in range(len(self.fault_names)):
            class_features = final_features[final_labels == i]
            if len(class_features) > 0:
                centroid = np.mean(class_features, axis=0)
                centroids.append(centroid)
            else:
                centroids.append(np.zeros(final_features.shape[1]))

        centroids = np.array(centroids)

        # Calculate distance matrix
        n_classes = len(self.fault_names)
        distance_matrix = np.zeros((n_classes, n_classes))

        for i in range(n_classes):
            for j in range(n_classes):
                distance_matrix[i, j] = np.linalg.norm(centroids[i] - centroids[j])

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(distance_matrix, cmap='YlOrRd')

        # Add labels
        ax.set_xticks(np.arange(n_classes))
        ax.set_yticks(np.arange(n_classes))
        ax.set_xticklabels(self.fault_names)
        ax.set_yticklabels(self.fault_names)

        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add text annotations
        for i in range(n_classes):
            for j in range(n_classes):
                text = ax.text(j, i, f'{distance_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=10)

        ax.set_title("Inter-class Distance Matrix (Multi-channel)", fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Euclidean Distance')
        plt.tight_layout()

        return fig

    def analyze_classification_performance(self):
        """Analyze classification performance with confusion matrix"""
        # Extract predictions
        final_features, final_labels, predictions = self.extract_features(
            self.final_model, self.val_loader, 'final'
        )

        # Create confusion matrix
        cm = confusion_matrix(final_labels, predictions)

        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create heatmap
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)

        # Add labels
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=self.fault_names,
               yticklabels=self.fault_names,
               title='Confusion Matrix (Multi-channel)',
               ylabel='True Label',
               xlabel='Predicted Label')

        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=12, fontweight='bold')

        plt.tight_layout()

        return fig, final_labels, predictions

    def analyze_feature_importance(self):
        """Analyze feature importance using Random Forest"""
        # Extract features and labels
        final_features, final_labels, _ = self.extract_features(
            self.final_model, self.val_loader, 'final'
        )

        # Train Random Forest for feature importance analysis
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(final_features, final_labels)

        # Get feature importance
        importance = rf.feature_importances_

        # Select top 20 features
        top_indices = np.argsort(importance)[-20:]
        top_importance = importance[top_indices]

        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 8))

        y_pos = np.arange(len(top_indices))
        bars = ax.barh(y_pos, top_importance, color='skyblue', alpha=0.8)

        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{width:.3f}', ha='left', va='center', fontsize=9)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([f'Feature {i}' for i in top_indices])
        ax.set_xlabel('Feature Importance')
        ax.set_title('Top 20 Feature Importance (Multi-channel)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        return fig

    def create_architecture_analysis(self):
        """Create model architecture analysis visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Model parameter distribution
        params = []
        param_names = []
        for name, param in self.final_model.named_parameters():
            if param.requires_grad:
                params.extend(param.data.cpu().flatten().numpy())
                param_names.append(name)

        ax1.hist(params, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
        ax1.set_title('Model Parameter Distribution', fontweight='bold')
        ax1.set_xlabel('Parameter Value')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)

        # Layer-wise parameter count
        layer_params = []
        layer_names = []
        for name, param in self.final_model.named_parameters():
            if param.requires_grad:
                layer_params.append(param.numel())
                layer_names.append(name.split('.')[0] if '.' in name else name)

        # Aggregate by layer type
        layer_counts = {}
        for name, count in zip(layer_names, layer_params):
            if name in layer_counts:
                layer_counts[name] += count
            else:
                layer_counts[name] = count

        names = list(layer_counts.keys())
        counts = list(layer_counts.values())

        bars = ax2.bar(range(len(names)), counts, color='lightcoral', alpha=0.8)
        ax2.set_title('Parameter Count by Layer Type', fontweight='bold')
        ax2.set_xlabel('Layer Type')
        ax2.set_ylabel('Parameter Count')
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=45)
        ax2.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                    f'{count:,}', ha='center', va='bottom', fontsize=9)

        # Multi-channel data distribution
        samples_per_channel = len(self.val_dataset) // 3  # Assuming 3 channels
        channels = ['DE (Drive End)', 'FE (Fan End)', 'BA (Base)']
        channel_counts = [samples_per_channel] * 3

        pie = ax3.pie(channel_counts, labels=channels, autopct='%1.1f%%',
                     colors=['#ff9999', '#66b3ff', '#99ff99'], startangle=90)
        ax3.set_title('Multi-channel Data Distribution', fontweight='bold')

        # Model architecture summary
        architecture_info = [
            'LSTM Encoder Architecture:',
            '• Input: Multi-channel signals (DE+FE+BA)',
            '• LSTM: Bidirectional, 2 layers, 128 hidden units',
            '• Attention: Self-attention mechanism',
            '• Feature Mapper: 256-dimensional features',
            '',
            'Classification Head:',
            '• Dropout: 0.3 regularization',
            '• Output: 4-class fault diagnosis',
            '',
            'Training Strategy:',
            '• Stage 1: Supervised Contrastive Learning',
            '• Stage 2: Transfer Learning Fine-tuning',
            '• Multi-channel data fusion (3x data augmentation)'
        ]

        ax4.text(0.05, 0.95, '\n'.join(architecture_info), transform=ax4.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Model Architecture Summary', fontweight='bold')

        plt.suptitle('Ex-ante Explainability: Model Architecture Analysis',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()

        return fig

    def generate_comprehensive_pdf_report(self):
        """Generate comprehensive PDF report with all analyses"""
        pdf_path = os.path.join(self.args.output_dir, 'comprehensive_explainability_report.pdf')

        with PdfPages(pdf_path) as pdf:
            # Title page
            fig_title = plt.figure(figsize=(11, 8.5))
            fig_title.text(0.5, 0.7, 'Multi-channel Bearing Fault Diagnosis',
                          ha='center', va='center', fontsize=24, fontweight='bold')
            fig_title.text(0.5, 0.6, 'Explainability Analysis Report',
                          ha='center', va='center', fontsize=20, fontweight='bold')
            fig_title.text(0.5, 0.5, 'Based on Supervised Contrastive Transfer Learning (SCTL-FD)',
                          ha='center', va='center', fontsize=14)
            fig_title.text(0.5, 0.4, f'Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}',
                          ha='center', va='center', fontsize=12)
            fig_title.text(0.5, 0.3, 'Multi-channel Data: DE + FE + BA Sensors',
                          ha='center', va='center', fontsize=12, style='italic')
            plt.axis('off')
            pdf.savefig(fig_title, bbox_inches='tight')
            plt.close(fig_title)

            # Ex-ante explainability
            print("Generating architecture analysis...")
            fig_arch = self.create_architecture_analysis()
            pdf.savefig(fig_arch, bbox_inches='tight')
            plt.close(fig_arch)

            # Transfer process explainability
            print("Generating feature transfer analysis...")
            fig_transfer = self.analyze_feature_transfer()
            pdf.savefig(fig_transfer, bbox_inches='tight')
            plt.close(fig_transfer)

            # Inter-class distance analysis
            print("Generating inter-class distance analysis...")
            fig_distance = self.analyze_inter_class_distances()
            pdf.savefig(fig_distance, bbox_inches='tight')
            plt.close(fig_distance)

            # Ex-post explainability
            print("Generating classification performance analysis...")
            fig_cm, true_labels, predictions = self.analyze_classification_performance()
            pdf.savefig(fig_cm, bbox_inches='tight')
            plt.close(fig_cm)

            # Feature importance analysis
            print("Generating feature importance analysis...")
            fig_importance = self.analyze_feature_importance()
            pdf.savefig(fig_importance, bbox_inches='tight')
            plt.close(fig_importance)

            # Performance summary page
            fig_summary = plt.figure(figsize=(11, 8.5))

            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

            accuracy = accuracy_score(true_labels, predictions)
            precision, recall, f1, support = precision_recall_fscore_support(true_labels, predictions, average='weighted')

            # Generate detailed classification report
            report = classification_report(true_labels, predictions, target_names=self.fault_names, digits=4)

            summary_text = f"""
MULTI-CHANNEL BEARING FAULT DIAGNOSIS PERFORMANCE SUMMARY

Overall Performance Metrics:
• Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)
• Weighted Precision: {precision:.4f}
• Weighted Recall: {recall:.4f}
• Weighted F1-Score: {f1:.4f}

Multi-channel Data Enhancement:
• Total Samples: {len(self.val_dataset):,}
• Channels Used: DE (Drive End) + FE (Fan End) + BA (Base)
• Data Augmentation: 3x increase from single-channel
• Fault Classes: {len(self.fault_names)} classes

Training Strategy:
• Stage 1: Supervised Contrastive Learning (SupCon)
• Stage 2: Transfer Learning Fine-tuning
• Architecture: LSTM + Attention + Multi-channel Fusion

Key Advantages:
• Multi-sensor redundancy for robust diagnosis
• Enhanced feature representation through channel fusion
• Superior generalization capability
• Comprehensive explainability analysis

Detailed Classification Report:
{report}

Model Architecture Highlights:
• Input: Multi-channel vibration signals (2048 samples)
• Encoder: Bidirectional LSTM (2 layers, 256 features)
• Attention: Self-attention mechanism for temporal focus
• Output: 4-class fault diagnosis with confidence scores
"""

            ax = fig_summary.add_subplot(111)
            ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.3))
            ax.axis('off')
            plt.axis('off')
            pdf.savefig(fig_summary, bbox_inches='tight')
            plt.close(fig_summary)

        print(f"✓ Comprehensive PDF report generated: {pdf_path}")
        return pdf_path

    def run_analysis(self):
        """Run complete explainability analysis"""
        print("Starting SCTL-FD Explainability Analysis (Multi-channel Data)...")

        # Load models and data
        self.load_models()
        self.load_data()

        # Generate comprehensive PDF report
        pdf_path = self.generate_comprehensive_pdf_report()

        print(f"🎉 Multi-channel Explainability Analysis Complete!")
        print(f"📊 Comprehensive PDF report saved at: {pdf_path}")

        return pdf_path

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='English Explainability Analysis with PDF Report')

    # Data parameters
    parser.add_argument('--val_data_path', type=str,
                        default='processed_data_all_channels/source_val_all_channels.npz',
                        help='Validation data path')

    # Model parameters
    parser.add_argument('--pretrained_encoder_path', type=str,
                        default='models_saved/lstm_supcon_all_channels/best_lstm_encoder.pth',
                        help='Pretrained encoder path')
    parser.add_argument('--final_model_path', type=str,
                        default='models_saved/lstm_transfer_all_channels/best_lstm_transfer_full_model.pth',
                        help='Final model path')

    # Model architecture parameters
    parser.add_argument('--signal_length', type=int, default=2048, help='Signal length')
    parser.add_argument('--feature_dim', type=int, default=256, help='Feature dimension')
    parser.add_argument('--hidden_dim', type=int, default=128, help='LSTM hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='LSTM layers')
    parser.add_argument('--num_classes', type=int, default=4, help='Number of classes')

    # Analysis parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--output_dir', type=str, default='results/english_explainability',
                        help='Output directory')

    args = parser.parse_args()

    # Print configuration
    print("English Explainability Analysis Configuration:")
    print("=" * 60)
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("=" * 60)

    # Run analysis
    analyzer = EnglishExplainabilityAnalyzer(args)
    analyzer.run_analysis()

if __name__ == '__main__':
    main()