# Machine Learning with Python: Weather Conditions and Climate Change with ClimateWins

A comprehensive machine learning project analyzing European weather patterns to predict climate consequences and extreme weather events using historical data spanning over a century. This project demonstrates advanced analytical capabilities from basic supervised learning through cutting-edge deep learning applications for climate science.

## Table of Contents
- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Data](#data)
- [Project Phases](#project-phases)
- [Key Findings](#key-findings)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [Contact](#contact)

## Project Overview

**Client**: ClimateWins - European nonprofit organization focused on climate research and prediction  
**Role**: Data Analyst with machine learning responsibilities  
**Objective**: Develop optimal machine learning models for predicting climate consequences and extreme weather events across Europe

This project addresses the growing need for accurate climate prediction tools in the context of increasing extreme weather events. By leveraging machine learning techniques on historical data, it provides actionable insights for climate adaptation and disaster preparedness strategies.

## Business Problem

ClimateWins faces critical challenges in predicting and preparing for climate change impacts:

- **Extreme Weather Increase**: 10-20 year trend of increasing severe weather events across Europe
- **Resource Constraints**: Limited budget requiring targeted, efficient prediction models
- **Geographic Variability**: Different climate patterns across European regions requiring location-specific approaches
- **Long-term Planning**: Need for 25-50 year climate projections to guide policy and safety recommendations

**Key Questions Addressed**:
- Can machine learning accurately predict pleasant vs. unpleasant weather conditions?
- What are the optimal algorithms for weather pattern recognition?
- How do geographic factors affect model performance?
- What patterns exist in 60+ years of European climate data?

## Data

**Source**: European Climate Assessment & Data Set project  
**Coverage**: 18 weather stations across Europe  
**Time Span**: Late 1800s to 2022 (120+ years of historical data)  
**Volume**: 16.6MB CSV with daily recordings  
**Key Variables**: Temperature, wind speed, snow depth, global radiation, atmospheric pressure

**Data Quality**: 
- Near-daily coverage for most recent decades
- Some historical gaps in early 1900s data
- Standardized measurement protocols across stations
- Geographic diversity spanning multiple climate zones

## Project Phases

### Phase 1: Supervised Learning Analysis ✅ COMPLETED
**Objective**: Identify optimal supervised learning model for weather prediction

**Methodology**:
- Implemented and compared K-Nearest Neighbor, Decision Trees, and Artificial Neural Networks
- Tested both multi-city and location-specific training approaches
- Applied gradient descent optimization for temperature trend analysis
- Evaluated model performance using accuracy metrics and confusion matrices

**Key Discovery**: Location-specific training improves model accuracy by 35-38% compared to multi-city approaches

### Phase 2: Advanced Machine Learning 🔄 IN PROGRESS
**Objective**: Apply advanced ML techniques for pattern discovery and long-term prediction

**Planned Implementation**:
- **Unsupervised Learning**: Clustering and dimensionality reduction to identify hidden weather patterns
- **Deep Learning**: CNNs and RNNs for complex temporal pattern recognition
- **Ensemble Methods**: Random forests and support vector machines for robust predictions
- **Generative Models**: GANs for future climate scenario generation
- **Hyperparameter Optimization**: Advanced tuning for maximum model performance

## Key Findings

### Geographic Specificity is Critical
All models performed **35-38% better** when trained on individual city data rather than multi-city datasets, revealing that:
- Weather patterns vary significantly across European cities
- Location-specific models are essential for accurate predictions
- Geographic factors must be considered in climate modeling approaches

### Model Performance Comparison

| Model | Multi-City Accuracy | Belgrade-Specific Accuracy | Improvement |
|-------|-------------------|--------------------------|-------------|
| **K-Nearest Neighbor** | 50% | **86%** | +36% |
| Decision Tree | 46% | 84% | +38% |
| Neural Network | 46% | 81% | +35% |

### Strategic Recommendations
- **Primary Choice**: K-Nearest Neighbor for single-variable weather predictions
- **Future Application**: Neural networks for multivariate analysis incorporating multiple weather variables
- **Implementation Strategy**: Deploy location-specific models rather than region-wide approaches

## Technologies Used

**Programming & Analysis**:
- Python 3.8+
- pandas, numpy for data manipulation
- scikit-learn for machine learning algorithms
- matplotlib, seaborn for visualization

**Advanced ML (Phase 2)**:
- TensorFlow/Keras for deep learning
- OpenCV for image processing applications
- scipy for statistical analysis

**Development Tools**:
- Jupyter Notebooks for analysis
- Git for version control
- Virtual environments for dependency management

## Project Structure

```
ClimateWins-ML-Project/
├── 01 Project Management/
│   ├── 01 Project Brief.pdf
│   └── 02 A2 Project Brief.pdf
├── 02 Data/
│   ├── Original Data/
│   │   ├── Answers-Weather_Prediction_Pleasant_Weather.csv
│   │   └── Dataset-weather-prediction-dataset-processed.csv
│   ├── Prepared Data/
│   │   └── weather_belgrade.csv
│   ├── Supervised/
│   └── Unsupervised/
├── 03 Scripts/
│   ├── Example Scripts/
│   │   ├── 1.3-Gradient-Descent-Example.ipynb
│   │   ├── 1.3-Gradient-Descent-for-Temperatures.ipynb
│   │   ├── 1.3-Simulated-Annealing-Example.ipynb
│   │   ├── 1.5-Iris-ANN.ipynb
│   │   └── 1.5-Iris-Decision-Tree.ipynb
│   └── Project Scripts/
│       ├── 01 Scaling Data for ML Analysis.ipynb
│       ├── 02 Gradient Descent.ipynb
│       ├── 03 KNN.ipynb
│       ├── 04 Decision Tree.ipynb
│       ├── 04 Decision Tree V2.ipynb
│       ├── 05 ANN Belgrade.ipynb
│       └── 06 Artificial Neural Network.ipynb
├── 04 Analysis/
│   ├── KNN confusion matrix.png
│   └── Task 1.3 Optimization Visualizations/
├── 05 Sent to Client/
│   ├── ClimateWins Interim ML Report.pdf
│   └── ClimateWins Interim ML Report.pptx
└── README.md
```

## Installation

### Prerequisites
```bash
Python 3.8+
Git
```

### Setup Instructions

1. **Clone the repository**:
```bash
git clone https://github.com/Sam-Abrams/ClimateWins-ML-Project.git
cd ClimateWins-ML-Project
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter tensorflow
```

4. **Download data** (if not included):
   - Visit [European Climate Assessment & Data Set project](https://www.ecad.eu/)
   - Download temperature dataset
   - Place in `02 Data/Original Data/` directory

## Usage

### Phase 1 Analysis (Completed)
```python
# Run the supervised learning analysis
jupyter notebook "03 Scripts/Project Scripts/03 KNN.ipynb"
jupyter notebook "03 Scripts/Project Scripts/04 Decision Tree.ipynb"
jupyter notebook "03 Scripts/Project Scripts/05 ANN Belgrade.ipynb"
```

### Data Exploration
```python
# Load and explore the dataset
import pandas as pd
data = pd.read_csv('02 Data/Original Data/Dataset-weather-prediction-dataset-processed.csv')
print(data.head())
print(data.describe())
```

### Model Training Example
```python
# Train KNN model on Belgrade data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Load Belgrade-specific data
belgrade_data = pd.read_csv('02 Data/Prepared Data/weather_belgrade.csv')

# Prepare features and target
X = belgrade_data[['Mean_temp']]
y = belgrade_data['pleasant']

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)

# Evaluate
accuracy = knn.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2%}")
```

## Results

### Phase 1 Achievements
- **Best Model Identified**: K-Nearest Neighbor with 86% accuracy on location-specific data
- **Critical Insight**: Geographic specificity dramatically improves prediction accuracy
- **Model Validation**: Comprehensive confusion matrix analysis across all tested algorithms
- **Optimization Success**: Gradient descent effectively identified temperature behavior patterns

### Business Impact
- **Prediction Capability**: Can accurately forecast pleasant weather conditions 86% of the time for specific locations
- **Resource Optimization**: Identified most efficient algorithm for ClimateWins' computational constraints
- **Strategic Direction**: Provided clear recommendation for location-specific model deployment

### Technical Demonstrations
- **Data Preprocessing**: Cleaned and standardized 120+ years of weather data
- **Model Comparison**: Systematic evaluation of multiple ML algorithms
- **Optimization Techniques**: Applied gradient descent for parameter optimization
- **Statistical Validation**: Robust testing methodology with proper train/test splits

## Future Work

### Phase 2 Implementation Roadmap

**Immediate Next Steps**:
- Implement unsupervised learning for pattern discovery in 60+ years of data
- Deploy CNNs and RNNs for complex temporal weather pattern recognition
- Apply hyperparameter tuning to optimize all models

**Advanced Applications**:
- Generate 25-50 year climate projections using ensemble methods
- Identify safest European locations for climate migration planning
- Develop real-time extreme weather prediction capabilities
- Create interactive dashboards for policy maker decision support

**Research Extensions**:
- Incorporate additional meteorological variables (humidity, pressure, wind patterns)
- Expand to global climate data integration
- Develop early warning systems for extreme weather events
- Investigate climate change acceleration patterns

## Academic and Professional Context

This project demonstrates proficiency in:
- **Statistical Analysis**: Hypothesis testing, model validation, performance metrics
- **Machine Learning**: Supervised learning, deep learning, optimization techniques
- **Data Science Workflow**: End-to-end project management from data collection to client delivery
- **Business Communication**: Translating complex technical findings into actionable business insights
- **Climate Science Applications**: Understanding meteorological data and climate modeling challenges

## Contributing

This project welcomes contributions, particularly in:
- Additional weather station data integration
- Alternative ML algorithm implementations
- Visualization improvements
- Climate science domain expertise

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**Sam Abrams** - Data Analyst & Machine Learning Specialist
- **Email**: sabrams15@gmail.com
- **Portfolio**: [www.sam-abrams.com](https://www.sam-abrams.com)
- **LinkedIn**: [Connect with me](https://linkedin.com/in/sam-abrams-profile)

## Acknowledgments

- **ClimateWins Organization** for project sponsorship and domain expertise
- **European Climate Assessment & Data Set project** for comprehensive weather data
- **CareerFoundry** for educational framework and mentorship

---

*This project showcases the application of machine learning techniques to climate science, demonstrating both technical proficiency and practical problem-solving skills for real-world environmental challenges.*
