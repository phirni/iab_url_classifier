https://huggingface.co/spaces/phirni/url_iab_classifier


#  IAB URL Classifier

An AI-powered web application that classifies webpage content into IAB (Interactive Advertising Bureau) categories using a fine-tuned DistilBERT model.

##  Features

- **Advanced Text Extraction**: Uses trafilatura with BeautifulSoup backup
- **32 IAB Categories**: Complete IAB taxonomy classification
- **Optimized Thresholds**: Category-specific confidence thresholds
- **Multi-label Classification**: Detects multiple relevant categories
- **Real-time Analysis**: Fast webpage content analysis

##  IAB Categories

The model classifies content into 32 standard IAB categories including:

- **Business**: Business And Finance, Careers, Law
- **Entertainment**: Entertainment, Sports, Video Gaming
- **Lifestyle**: Food & Drink, Travel, Style & Fashion
- **Health**: Medical Health, Healthy Living
- **Technology**: Technology & Computing, Science
- **And many more...**

##  Technical Details

- **Model**: Fine-tuned DistilBERT (phirni/iab-url-classifier)
- **Architecture**: Multi-label sequence classification
- **Extraction**: Trafilatura + BeautifulSoup fallback
- **Optimization**: Per-category confidence thresholds
- **Performance**: 72% F1-score with optimized thresholds

##  Model Performance

- **Exact Match Accuracy**: 47.4%
- **F1-Score (Macro)**: 72.0%
- **Jaccard Score**: 65.7%
- **Optimized Thresholds**: Category-specific for best performance

##  Usage

1. Enter any webpage URL
2. Click "Analyze Content"
3. View predicted IAB categories with confidence scores
4. Get detailed breakdown of content classification

Perfect for content categorization, ad targeting, SEO analysis, and content strategy.
