# 🎨 UX Integration Guide

*Comprehensive guide for UX designers and researchers integrating SAMO Brain's emotion detection capabilities into design processes, user research, and prototyping workflows.*

## 🚀 Quick Start

### Test Emotion Detection in Your Design Process

```javascript
// Quick emotion analysis for design validation
const testEmotionAnalysis = async (userFeedback) => {
  try {
    const response = await fetch('http://localhost:8000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: userFeedback })
    });
    
    const result = await response.json();
    console.log('User Emotion:', result.predicted_emotion);
    console.log('Confidence:', result.confidence);
    
    return result;
  } catch (error) {
    console.error('Emotion analysis failed:', error);
  }
};

// Test with sample user feedback
testEmotionAnalysis("I'm really frustrated with this interface - nothing works as expected!");
```

### Emotion-Aware Design System

```css
/* Emotion-based color system for consistent UX */
:root {
  /* Primary emotion colors */
  --emotion-happy: #FFD54F;
  --emotion-sad: #7986CB;
  --emotion-excited: #FFA726;
  --emotion-calm: #4ECDC4;
  --emotion-frustrated: #FF7043;
  --emotion-anxious: #FF6B6B;
  --emotion-grateful: #66BB6A;
  --emotion-hopeful: #81C784;
  --emotion-overwhelmed: #9575CD;
  --emotion-proud: #4DB6AC;
  --emotion-content: #45B7D1;
  --emotion-tired: #A1887F;
  
  /* Emotion intensity variants */
  --emotion-light: 0.1;
  --emotion-medium: 0.3;
  --emotion-strong: 0.6;
}

/* Emotion-aware component styles */
.emotion-card {
  border-radius: 12px;
  padding: 16px;
  transition: all 0.3s ease;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.emotion-card.happy {
  background: linear-gradient(135deg, var(--emotion-happy), #FFF9C4);
  border-left: 4px solid var(--emotion-happy);
}

.emotion-card.frustrated {
  background: linear-gradient(135deg, var(--emotion-frustrated), #FFCCBC);
  border-left: 4px solid var(--emotion-frustrated);
}

.emotion-card.calm {
  background: linear-gradient(135deg, var(--emotion-calm), #B2EBF2);
  border-left: 4px solid var(--emotion-calm);
}
```

## 🎯 Design System Integration

### Emotion-Aware Component Library

```jsx
// React component for emotion-aware UI elements
import React from 'react';
import './EmotionAwareComponents.css';

const EmotionAwareButton = ({ 
  emotion, 
  children, 
  onClick, 
  variant = 'primary' 
}) => {
  const getEmotionStyles = (emotion) => {
    const emotionStyles = {
      happy: {
        background: 'var(--emotion-happy)',
        color: '#000',
        animation: 'pulse 2s infinite'
      },
      sad: {
        background: 'var(--emotion-sad)',
        color: '#fff',
        animation: 'fadeIn 0.5s ease'
      },
      excited: {
        background: 'var(--emotion-excited)',
        color: '#000',
        animation: 'bounce 1s infinite'
      },
      calm: {
        background: 'var(--emotion-calm)',
        color: '#000',
        animation: 'none'
      },
      frustrated: {
        background: 'var(--emotion-frustrated)',
        color: '#fff',
        animation: 'shake 0.5s ease'
      }
    };
    
    return emotionStyles[emotion] || emotionStyles.calm;
  };

  return (
    <button
      className={`emotion-button ${variant}`}
      style={getEmotionStyles(emotion)}
      onClick={onClick}
    >
      {children}
    </button>
  );
};

const EmotionAwareCard = ({ emotion, children, title }) => {
  return (
    <div className={`emotion-card ${emotion}`}>
      <div className="emotion-indicator">
        <span className="emotion-icon">
          {getEmotionIcon(emotion)}
        </span>
        <span className="emotion-label">{emotion}</span>
      </div>
      <h3>{title}</h3>
      <div className="card-content">
        {children}
      </div>
    </div>
  );
};

const getEmotionIcon = (emotion) => {
  const icons = {
    happy: '😄',
    sad: '😢',
    excited: '🤩',
    calm: '😌',
    frustrated: '😤',
    anxious: '😰',
    grateful: '🙏',
    hopeful: '🤗',
    overwhelmed: '😵',
    proud: '😎',
    content: '😊',
    tired: '😴'
  };
  return icons[emotion] || '😐';
};

export { EmotionAwareButton, EmotionAwareCard };
```

### Emotion-Based Typography System

```css
/* Emotion-aware typography system */
.emotion-text {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  transition: all 0.3s ease;
}

.emotion-text.happy {
  font-weight: 600;
  color: var(--emotion-happy);
  text-shadow: 0 1px 2px rgba(255, 213, 79, 0.3);
}

.emotion-text.sad {
  font-weight: 400;
  color: var(--emotion-sad);
  font-style: italic;
}

.emotion-text.excited {
  font-weight: 700;
  color: var(--emotion-excited);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.emotion-text.calm {
  font-weight: 300;
  color: var(--emotion-calm);
  line-height: 1.6;
}

.emotion-text.frustrated {
  font-weight: 600;
  color: var(--emotion-frustrated);
  text-decoration: underline;
}
```

## 🔬 User Research Integration

### Emotion-Aware User Testing Framework

```python
import requests
import pandas as pd
from datetime import datetime
import json

class EmotionAwareUserTesting:
    def __init__(self, api_base_url="http://localhost:8000"):
        self.api_base_url = api_base_url
        self.test_sessions = []
    
    def record_user_feedback(self, session_id, user_id, feedback_text, 
                           task_completed, task_difficulty, ui_element):
        """Record user feedback with emotion analysis."""
        try:
            # Analyze emotion from feedback
            emotion_response = requests.post(
                f"{self.api_base_url}/predict",
                json={'text': feedback_text}
            )
            emotion_result = emotion_response.json()
            
            # Record session data
            session_data = {
                'session_id': session_id,
                'user_id': user_id,
                'timestamp': datetime.now().isoformat(),
                'feedback_text': feedback_text,
                'predicted_emotion': emotion_result['predicted_emotion'],
                'emotion_confidence': emotion_result['confidence'],
                'task_completed': task_completed,
                'task_difficulty': task_difficulty,
                'ui_element': ui_element,
                'all_emotions': emotion_result['probabilities']
            }
            
            self.test_sessions.append(session_data)
            return session_data
            
        except Exception as e:
            print(f"Error recording user feedback: {e}")
            return None
    
    def analyze_user_sentiment_trends(self):
        """Analyze sentiment trends across user testing sessions."""
        if not self.test_sessions:
            return "No test sessions recorded"
        
        df = pd.DataFrame(self.test_sessions)
        
        # Emotion distribution analysis
        emotion_distribution = df['predicted_emotion'].value_counts()
        
        # Task completion vs emotion correlation
        completion_by_emotion = df.groupby('predicted_emotion')['task_completed'].agg(['mean', 'count'])
        
        # Difficulty vs emotion correlation
        difficulty_by_emotion = df.groupby('predicted_emotion')['task_difficulty'].mean()
        
        # UI element emotion analysis
        ui_emotion_analysis = df.groupby('ui_element')['predicted_emotion'].value_counts()
        
        return {
            'emotion_distribution': emotion_distribution.to_dict(),
            'completion_by_emotion': completion_by_emotion.to_dict(),
            'difficulty_by_emotion': difficulty_by_emotion.to_dict(),
            'ui_emotion_analysis': ui_emotion_analysis.to_dict(),
            'total_sessions': len(df),
            'unique_users': df['user_id'].nunique()
        }
    
    def generate_ux_insights_report(self):
        """Generate actionable UX insights from emotion data."""
        analysis = self.analyze_user_sentiment_trends()
        
        insights = {
            'timestamp': datetime.now().isoformat(),
            'key_findings': [],
            'recommendations': [],
            'high_priority_issues': []
        }
        
        # Analyze negative emotions
        negative_emotions = ['frustrated', 'anxious', 'overwhelmed', 'sad']
        negative_sessions = [s for s in self.test_sessions 
                           if s['predicted_emotion'] in negative_emotions]
        
        if negative_sessions:
            insights['key_findings'].append({
                'type': 'negative_emotions_detected',
                'count': len(negative_sessions),
                'percentage': (len(negative_sessions) / len(self.test_sessions)) * 100,
                'emotions': [s['predicted_emotion'] for s in negative_sessions]
            })
            
            # Identify problematic UI elements
            problematic_elements = {}
            for session in negative_sessions:
                element = session['ui_element']
                if element not in problematic_elements:
                    problematic_elements[element] = []
                problematic_elements[element].append(session['predicted_emotion'])
            
            for element, emotions in problematic_elements.items():
                insights['high_priority_issues'].append({
                    'ui_element': element,
                    'negative_emotions': emotions,
                    'recommendation': f'Redesign {element} to reduce user frustration'
                })
        
        # Analyze positive emotions
        positive_emotions = ['happy', 'excited', 'grateful', 'proud', 'calm']
        positive_sessions = [s for s in self.test_sessions 
                           if s['predicted_emotion'] in positive_emotions]
        
        if positive_sessions:
            insights['key_findings'].append({
                'type': 'positive_emotions_detected',
                'count': len(positive_sessions),
                'percentage': (len(positive_sessions) / len(self.test_sessions)) * 100,
                'emotions': [s['predicted_emotion'] for s in positive_sessions]
            })
        
        return insights
    
    def export_test_data(self, format='csv'):
        """Export test data for further analysis."""
        if format == 'csv':
            df = pd.DataFrame(self.test_sessions)
            filename = f"user_testing_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False)
            return filename
        elif format == 'json':
            filename = f"user_testing_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(self.test_sessions, f, indent=2)
            return filename

# Usage example
ux_testing = EmotionAwareUserTesting()

# Record user feedback during testing
ux_testing.record_user_feedback(
    session_id="test_001",
    user_id="user_123",
    feedback_text="This interface is so confusing, I can't find anything!",
    task_completed=False,
    task_difficulty=5,
    ui_element="navigation_menu"
)

# Generate insights report
insights = ux_testing.generate_ux_insights_report()
print(json.dumps(insights, indent=2))
```

### A/B Testing with Emotion Analysis

```python
class EmotionAwareABTesting:
    def __init__(self, api_base_url="http://localhost:8000"):
        self.api_base_url = api_base_url
        self.variant_results = {}
    
    def test_variant_emotion_response(self, variant_name, user_feedback_list):
        """Test how different UI variants affect user emotions."""
        variant_data = []
        
        for feedback in user_feedback_list:
            try:
                emotion_response = requests.post(
                    f"{self.api_base_url}/predict",
                    json={'text': feedback}
                )
                emotion_result = emotion_response.json()
                
                variant_data.append({
                    'feedback': feedback,
                    'emotion': emotion_result['predicted_emotion'],
                    'confidence': emotion_result['confidence'],
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                print(f"Error analyzing feedback: {e}")
        
        self.variant_results[variant_name] = variant_data
        return variant_data
    
    def compare_variants(self, variant_a, variant_b):
        """Compare emotion responses between two UI variants."""
        if variant_a not in self.variant_results or variant_b not in self.variant_results:
            return "Both variants must be tested first"
        
        a_data = self.variant_results[variant_a]
        b_data = self.variant_results[variant_b]
        
        # Calculate emotion distributions
        a_emotions = [d['emotion'] for d in a_data]
        b_emotions = [d['emotion'] for d in b_data]
        
        a_emotion_dist = pd.Series(a_emotions).value_counts()
        b_emotion_dist = pd.Series(b_emotions).value_counts()
        
        # Calculate average confidence
        a_avg_confidence = np.mean([d['confidence'] for d in a_data])
        b_avg_confidence = np.mean([d['confidence'] for d in b_data])
        
        # Identify positive vs negative emotions
        positive_emotions = ['happy', 'excited', 'grateful', 'proud', 'calm', 'content']
        negative_emotions = ['frustrated', 'anxious', 'overwhelmed', 'sad', 'tired']
        
        a_positive_ratio = len([e for e in a_emotions if e in positive_emotions]) / len(a_emotions)
        b_positive_ratio = len([e for e in b_emotions if e in positive_emotions]) / len(b_emotions)
        
        comparison = {
            'variant_a': {
                'name': variant_a,
                'total_feedback': len(a_data),
                'emotion_distribution': a_emotion_dist.to_dict(),
                'avg_confidence': a_avg_confidence,
                'positive_emotion_ratio': a_positive_ratio
            },
            'variant_b': {
                'name': variant_b,
                'total_feedback': len(b_data),
                'emotion_distribution': b_emotion_dist.to_dict(),
                'avg_confidence': b_avg_confidence,
                'positive_emotion_ratio': b_positive_ratio
            },
            'recommendation': self._generate_recommendation(a_positive_ratio, b_positive_ratio)
        }
        
        return comparison
    
    def _generate_recommendation(self, ratio_a, ratio_b):
        """Generate recommendation based on emotion analysis."""
        if abs(ratio_a - ratio_b) < 0.1:
            return "Both variants perform similarly emotionally. Consider other factors."
        elif ratio_a > ratio_b:
            return f"Variant A generates more positive emotions. Consider implementing Variant A."
        else:
            return f"Variant B generates more positive emotions. Consider implementing Variant B."
```

## 🎨 Prototyping with Emotion Data

### Figma/Sketch Integration Patterns

```javascript
// Emotion-aware design tokens for design systems
const emotionDesignTokens = {
  colors: {
    happy: {
      primary: '#FFD54F',
      secondary: '#FFF9C4',
      accent: '#FFC107',
      text: '#000000'
    },
    sad: {
      primary: '#7986CB',
      secondary: '#E8EAF6',
      accent: '#3F51B5',
      text: '#FFFFFF'
    },
    excited: {
      primary: '#FFA726',
      secondary: '#FFF3E0',
      accent: '#FF9800',
      text: '#000000'
    },
    calm: {
      primary: '#4ECDC4',
      secondary: '#E0F2F1',
      accent: '#009688',
      text: '#000000'
    },
    frustrated: {
      primary: '#FF7043',
      secondary: '#FFCCBC',
      accent: '#FF5722',
      text: '#FFFFFF'
    }
  },
  
  typography: {
    happy: {
      fontFamily: 'Inter Bold',
      fontSize: '16px',
      lineHeight: '1.5',
      letterSpacing: '0.5px'
    },
    sad: {
      fontFamily: 'Inter Regular',
      fontSize: '14px',
      lineHeight: '1.6',
      fontStyle: 'italic'
    },
    excited: {
      fontFamily: 'Inter Black',
      fontSize: '18px',
      lineHeight: '1.4',
      textTransform: 'uppercase'
    }
  },
  
  spacing: {
    happy: {
      padding: '16px',
      margin: '8px',
      borderRadius: '12px'
    },
    sad: {
      padding: '12px',
      margin: '4px',
      borderRadius: '8px'
    },
    excited: {
      padding: '20px',
      margin: '12px',
      borderRadius: '16px'
    }
  }
};

// Export for design tools
export const getEmotionStyles = (emotion) => {
  return {
    colors: emotionDesignTokens.colors[emotion] || emotionDesignTokens.colors.calm,
    typography: emotionDesignTokens.typography[emotion] || emotionDesignTokens.typography.calm,
    spacing: emotionDesignTokens.spacing[emotion] || emotionDesignTokens.spacing.calm
  };
};
```

### Interactive Prototype with Emotion Feedback

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Emotion-Aware Prototype</title>
    <style>
        .prototype-container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            font-family: 'Inter', sans-serif;
        }
        
        .emotion-feedback {
            background: #f5f5f5;
            padding: 20px;
            border-radius: 12px;
            margin: 20px 0;
        }
        
        .emotion-visualization {
            display: flex;
            gap: 10px;
            margin: 10px 0;
        }
        
        .emotion-bar {
            flex: 1;
            height: 20px;
            border-radius: 10px;
            position: relative;
            overflow: hidden;
        }
        
        .emotion-fill {
            height: 100%;
            transition: width 0.3s ease;
        }
        
        .prototype-element {
            padding: 15px;
            margin: 10px 0;
            border: 2px solid #ddd;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .prototype-element:hover {
            border-color: #007bff;
            transform: translateY(-2px);
        }
    </style>
</head>
<body>
    <div class="prototype-container">
        <h1>🎨 Emotion-Aware UI Prototype</h1>
        
        <div class="emotion-feedback">
            <h3>User Emotion Analysis</h3>
            <textarea id="userFeedback" placeholder="Describe your experience with this interface..." rows="4" style="width: 100%; padding: 10px;"></textarea>
            <button onclick="analyzeEmotion()" style="margin-top: 10px; padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer;">
                Analyze Emotion
            </button>
            
            <div id="emotionResults" style="margin-top: 20px;"></div>
        </div>
        
        <div class="prototype-element" onclick="testElement('navigation')">
            <h3>Navigation Menu</h3>
            <p>Click to test user reaction to navigation design</p>
        </div>
        
        <div class="prototype-element" onclick="testElement('button')">
            <h3>Call-to-Action Button</h3>
            <p>Click to test user reaction to button design</p>
        </div>
        
        <div class="prototype-element" onclick="testElement('form')">
            <h3>Form Interface</h3>
            <p>Click to test user reaction to form design</p>
        </div>
    </div>

    <script>
        async function analyzeEmotion() {
            const feedback = document.getElementById('userFeedback').value;
            if (!feedback.trim()) {
                alert('Please enter some feedback first');
                return;
            }
            
            try {
                const response = await fetch('http://localhost:8000/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: feedback })
                });
                
                const result = await response.json();
                displayEmotionResults(result);
            } catch (error) {
                console.error('Error analyzing emotion:', error);
                alert('Error analyzing emotion. Please check if the API is running.');
            }
        }
        
        function displayEmotionResults(result) {
            const container = document.getElementById('emotionResults');
            
            const emotionColors = {
                happy: '#FFD54F', sad: '#7986CB', excited: '#FFA726',
                calm: '#4ECDC4', frustrated: '#FF7043', anxious: '#FF6B6B',
                grateful: '#66BB6A', hopeful: '#81C784', overwhelmed: '#9575CD',
                proud: '#4DB6AC', content: '#45B7D1', tired: '#A1887F'
            };
            
            const emotionIcons = {
                happy: '😄', sad: '😢', excited: '🤩', calm: '😌',
                frustrated: '😤', anxious: '😰', grateful: '🙏', hopeful: '🤗',
                overwhelmed: '😵', proud: '😎', content: '😊', tired: '😴'
            };
            
            let html = `
                <div style="background: white; padding: 15px; border-radius: 8px; margin-top: 15px;">
                    <h4>${emotionIcons[result.predicted_emotion]} Primary Emotion: ${result.predicted_emotion.charAt(0).toUpperCase() + result.predicted_emotion.slice(1)}</h4>
                    <p>Confidence: ${(result.confidence * 100).toFixed(1)}%</p>
                    
                    <h5>All Emotions Detected:</h5>
                    <div class="emotion-visualization">
            `;
            
            Object.entries(result.probabilities)
                .sort(([,a], [,b]) => b - a)
                .forEach(([emotion, probability]) => {
                    html += `
                        <div style="flex: 1; text-align: center;">
                            <div class="emotion-bar" style="background: #eee;">
                                <div class="emotion-fill" style="width: ${probability * 100}%; background: ${emotionColors[emotion]};"></div>
                            </div>
                            <div style="font-size: 12px; margin-top: 5px;">
                                ${emotionIcons[emotion]} ${(probability * 100).toFixed(1)}%
                            </div>
                        </div>
                    `;
                });
            
            html += '</div></div>';
            container.innerHTML = html;
        }
        
        function testElement(elementType) {
            const testPrompts = {
                navigation: "How do you feel about the navigation design?",
                button: "What's your reaction to this button design?",
                form: "How does this form make you feel?"
            };
            
            document.getElementById('userFeedback').value = testPrompts[elementType];
            analyzeEmotion();
        }
    </script>
</body>
</html>
```

## ♿ Accessibility Considerations

### Emotion-Aware Accessibility Patterns

```javascript
// Accessibility-enhanced emotion components
class AccessibleEmotionComponent {
    constructor(element, emotion) {
        this.element = element;
        this.emotion = emotion;
        this.setupAccessibility();
    }
    
    setupAccessibility() {
        // Add ARIA labels for screen readers
        this.element.setAttribute('aria-label', `Content with ${this.emotion} emotional context`);
        
        // Add role for semantic meaning
        this.element.setAttribute('role', 'region');
        
        // Add live region for dynamic updates
        this.element.setAttribute('aria-live', 'polite');
        
        // Add emotion-specific accessibility features
        this.addEmotionSpecificAccessibility();
    }
    
    addEmotionSpecificAccessibility() {
        const accessibilityFeatures = {
            happy: {
                ariaDescription: 'This content conveys positive, uplifting emotions',
                highContrast: true,
                reducedMotion: false
            },
            sad: {
                ariaDescription: 'This content may contain sensitive or emotional material',
                highContrast: false,
                reducedMotion: true
            },
            anxious: {
                ariaDescription: 'This content may cause anxiety or stress',
                highContrast: false,
                reducedMotion: true,
                warning: 'Content may be anxiety-inducing'
            },
            calm: {
                ariaDescription: 'This content is designed to be calming and peaceful',
                highContrast: false,
                reducedMotion: true
            }
        };
        
        const features = accessibilityFeatures[this.emotion] || accessibilityFeatures.calm;
        
        this.element.setAttribute('aria-description', features.ariaDescription);
        
        if (features.warning) {
            this.addWarningAnnouncement(features.warning);
        }
    }
    
    addWarningAnnouncement(warning) {
        const announcement = document.createElement('div');
        announcement.setAttribute('aria-live', 'assertive');
        announcement.setAttribute('role', 'alert');
        announcement.className = 'sr-only';
        announcement.textContent = warning;
        
        document.body.appendChild(announcement);
        
        // Remove after announcement
        setTimeout(() => {
            document.body.removeChild(announcement);
        }, 1000);
    }
    
    updateEmotion(newEmotion) {
        this.emotion = newEmotion;
        this.setupAccessibility();
    }
}

// CSS for screen reader only content
const styles = `
.sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
}

/* High contrast mode support */
@media (prefers-contrast: high) {
    .emotion-card {
        border: 2px solid currentColor;
    }
}

/* Reduced motion support */
@media (prefers-reduced-motion: reduce) {
    .emotion-card {
        animation: none;
        transition: none;
    }
}

/* Color blind friendly emotion indicators */
.emotion-indicator {
    display: flex;
    align-items: center;
    gap: 8px;
}

.emotion-indicator::before {
    content: '';
    width: 12px;
    height: 12px;
    border-radius: 50%;
    border: 2px solid currentColor;
}

.emotion-indicator.happy::before {
    background: var(--emotion-happy);
}

.emotion-indicator.sad::before {
    background: var(--emotion-sad);
}

.emotion-indicator.frustrated::before {
    background: var(--emotion-frustrated);
}
`;
```

## 🤝 Collaboration Workflows

### Design-Development Handoff with Emotion Data

```javascript
// Design system export with emotion specifications
const exportDesignSystem = () => {
    const designSystem = {
        version: '1.0.0',
        lastUpdated: new Date().toISOString(),
        emotions: {
            happy: {
                description: 'Positive, uplifting user experience',
                useCases: ['success states', 'achievements', 'positive feedback'],
                colors: {
                    primary: '#FFD54F',
                    secondary: '#FFF9C4',
                    text: '#000000'
                },
                typography: {
                    fontFamily: 'Inter Bold',
                    fontSize: '16px',
                    lineHeight: '1.5'
                },
                animations: {
                    duration: '0.3s',
                    easing: 'ease-out',
                    effects: ['pulse', 'bounce']
                },
                accessibility: {
                    ariaLabel: 'Positive emotion indicator',
                    highContrast: true,
                    reducedMotion: false
                }
            },
            sad: {
                description: 'Sensitive, empathetic user experience',
                useCases: ['error states', 'loss', 'sensitive content'],
                colors: {
                    primary: '#7986CB',
                    secondary: '#E8EAF6',
                    text: '#FFFFFF'
                },
                typography: {
                    fontFamily: 'Inter Regular',
                    fontSize: '14px',
                    lineHeight: '1.6',
                    fontStyle: 'italic'
                },
                animations: {
                    duration: '0.5s',
                    easing: 'ease-in',
                    effects: ['fadeIn']
                },
                accessibility: {
                    ariaLabel: 'Sensitive content indicator',
                    highContrast: false,
                    reducedMotion: true
                }
            }
        },
        
        components: {
            emotionButton: {
                variants: ['happy', 'sad', 'excited', 'calm', 'frustrated'],
                props: {
                    emotion: 'string',
                    size: 'small | medium | large',
                    variant: 'primary | secondary | ghost'
                },
                examples: {
                    happy: {
                        text: 'Great job!',
                        emotion: 'happy',
                        size: 'medium',
                        variant: 'primary'
                    },
                    sad: {
                        text: 'I understand',
                        emotion: 'sad',
                        size: 'medium',
                        variant: 'secondary'
                    }
                }
            },
            
            emotionCard: {
                variants: ['happy', 'sad', 'excited', 'calm', 'frustrated'],
                props: {
                    emotion: 'string',
                    title: 'string',
                    content: 'string',
                    showEmotionIndicator: 'boolean'
                }
            }
        }
    };
    
    return JSON.stringify(designSystem, null, 2);
};

// Generate design tokens for development
const generateDesignTokens = () => {
    const tokens = {
        colors: {
            emotion: {
                happy: {
                    value: '#FFD54F',
                    type: 'color'
                },
                sad: {
                    value: '#7986CB',
                    type: 'color'
                },
                excited: {
                    value: '#FFA726',
                    type: 'color'
                },
                calm: {
                    value: '#4ECDC4',
                    type: 'color'
                },
                frustrated: {
                    value: '#FF7043',
                    type: 'color'
                }
            }
        },
        
        typography: {
            emotion: {
                happy: {
                    fontFamily: { value: 'Inter Bold', type: 'fontFamily' },
                    fontSize: { value: '16px', type: 'fontSize' },
                    lineHeight: { value: '1.5', type: 'lineHeight' }
                },
                sad: {
                    fontFamily: { value: 'Inter Regular', type: 'fontFamily' },
                    fontSize: { value: '14px', type: 'fontSize' },
                    lineHeight: { value: '1.6', type: 'lineHeight' }
                }
            }
        },
        
        spacing: {
            emotion: {
                happy: {
                    padding: { value: '16px', type: 'spacing' },
                    margin: { value: '8px', type: 'spacing' }
                },
                sad: {
                    padding: { value: '12px', type: 'spacing' },
                    margin: { value: '4px', type: 'spacing' }
                }
            }
        }
    };
    
    return tokens;
};
```

### UX Research Documentation Template

```markdown
# UX Research Report Template

## Research Session Details
- **Date**: [Date]
- **Session ID**: [ID]
- **Participants**: [Number]
- **Research Method**: [Method]
- **Duration**: [Duration]

## Emotion Analysis Summary
- **Primary Emotions Detected**: [List]
- **Emotion Distribution**: [Chart/Data]
- **Confidence Levels**: [Average/Stats]

## Key Findings

### Positive Emotions
- **Emotion**: [Emotion]
- **Frequency**: [Number/Percentage]
- **Context**: [When/Why it occurred]
- **UI Elements**: [Associated elements]

### Negative Emotions
- **Emotion**: [Emotion]
- **Frequency**: [Number/Percentage]
- **Context**: [When/Why it occurred]
- **UI Elements**: [Associated elements]
- **Severity**: [Low/Medium/High]

## Design Recommendations

### Immediate Actions
- [ ] [Action item]
- [ ] [Action item]

### Future Considerations
- [ ] [Action item]
- [ ] [Action item]

## Technical Implementation Notes
- **API Endpoints Used**: [List]
- **Data Collection Method**: [Method]
- **Analysis Tools**: [Tools]

## Next Steps
1. [Next step]
2. [Next step]
3. [Next step]
```

## 📊 Analytics and Insights

### Emotion Analytics Dashboard

```python
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests

class EmotionAnalyticsDashboard:
    def __init__(self, api_base_url="http://localhost:8000"):
        self.api_base_url = api_base_url
    
    def run_dashboard(self):
        st.set_page_config(page_title="SAMO Brain - Emotion Analytics", layout="wide")
        
        st.title("🧠 SAMO Brain - Emotion Analytics Dashboard")
        st.markdown("Real-time emotion analysis and UX insights")
        
        # Sidebar filters
        st.sidebar.header("Filters")
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(datetime.now() - timedelta(days=7), datetime.now()),
            max_value=datetime.now()
        )
        
        emotion_filter = st.sidebar.multiselect(
            "Emotions",
            ['happy', 'sad', 'excited', 'calm', 'frustrated', 'anxious', 
             'grateful', 'hopeful', 'overwhelmed', 'proud', 'content', 'tired'],
            default=['happy', 'sad', 'frustrated']
        )
        
        # Main dashboard
        col1, col2 = st.columns(2)
        
        with col1:
            self.show_emotion_distribution()
        
        with col2:
            self.show_emotion_trends()
        
        col3, col4 = st.columns(2)
        
        with col3:
            self.show_ui_element_analysis()
        
        with col4:
            self.show_user_sentiment_flow()
        
        # Detailed analysis
        st.header("Detailed Analysis")
        self.show_detailed_insights()
    
    def show_emotion_distribution(self):
        st.subheader("📊 Emotion Distribution")
        
        # Mock data - replace with real API calls
        emotion_data = {
            'happy': 25, 'sad': 15, 'excited': 20, 'calm': 18,
            'frustrated': 12, 'anxious': 8, 'grateful': 10
        }
        
        df = pd.DataFrame(list(emotion_data.items()), columns=['Emotion', 'Count'])
        
        fig = px.pie(df, values='Count', names='Emotion', 
                    title="Emotion Distribution (Last 7 Days)")
        st.plotly_chart(fig, use_container_width=True)
    
    def show_emotion_trends(self):
        st.subheader("📈 Emotion Trends")
        
        # Mock time series data
        dates = pd.date_range(start=datetime.now() - timedelta(days=7), 
                            end=datetime.now(), freq='D')
        
        trend_data = {
            'happy': [20, 22, 25, 23, 26, 24, 25],
            'frustrated': [15, 12, 10, 8, 6, 5, 4],
            'calm': [18, 19, 20, 21, 22, 23, 24]
        }
        
        fig = go.Figure()
        for emotion, values in trend_data.items():
            fig.add_trace(go.Scatter(x=dates, y=values, name=emotion, mode='lines+markers'))
        
        fig.update_layout(title="Emotion Trends Over Time", xaxis_title="Date", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    
    def show_ui_element_analysis(self):
        st.subheader("🎨 UI Element Analysis")
        
        # Mock UI element data
        ui_data = {
            'Navigation': {'happy': 30, 'frustrated': 10, 'calm': 20},
            'Buttons': {'happy': 25, 'frustrated': 15, 'calm': 15},
            'Forms': {'happy': 15, 'frustrated': 25, 'calm': 10},
            'Content': {'happy': 35, 'frustrated': 5, 'calm': 25}
        }
        
        df = pd.DataFrame(ui_data).T
        fig = px.bar(df, title="Emotions by UI Element")
        st.plotly_chart(fig, use_container_width=True)
    
    def show_user_sentiment_flow(self):
        st.subheader("🔄 User Sentiment Flow")
        
        # Mock user journey data
        journey_data = {
            'Step': ['Landing', 'Navigation', 'Action', 'Completion'],
            'Positive': [80, 70, 85, 90],
            'Neutral': [15, 20, 10, 8],
            'Negative': [5, 10, 5, 2]
        }
        
        df = pd.DataFrame(journey_data)
        fig = px.line(df, x='Step', y=['Positive', 'Neutral', 'Negative'],
                     title="Sentiment Flow Through User Journey")
        st.plotly_chart(fig, use_container_width=True)
    
    def show_detailed_insights(self):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔍 Key Insights")
            
            insights = [
                "🎯 Navigation menu causes 25% of user frustration",
                "✅ Call-to-action buttons generate 85% positive emotions",
                "⚠️ Form fields trigger anxiety in 15% of users",
                "🎉 Success messages create 90% positive sentiment"
            ]
            
            for insight in insights:
                st.write(insight)
        
        with col2:
            st.subheader("📋 Recommendations")
            
            recommendations = [
                "🔧 Redesign navigation with clearer hierarchy",
                "🎨 Add micro-interactions to reduce form anxiety",
                "✨ Implement more positive feedback moments",
                "📱 Optimize mobile experience for better emotions"
            ]
            
            for rec in recommendations:
                st.write(rec)

# Run the dashboard
if __name__ == "__main__":
    dashboard = EmotionAnalyticsDashboard()
    dashboard.run_dashboard()
```

## 🚀 Getting Started Checklist

### For UX Designers
- [ ] **Set up emotion analysis API connection**
- [ ] **Install design system with emotion tokens**
- [ ] **Create emotion-aware component library**
- [ ] **Set up user testing with emotion tracking**
- [ ] **Configure accessibility features**
- [ ] **Test emotion visualization components**

### For UX Researchers
- [ ] **Configure emotion-aware user testing framework**
- [ ] **Set up A/B testing with emotion analysis**
- [ ] **Create research documentation templates**
- [ ] **Establish baseline emotion metrics**
- [ ] **Set up analytics dashboard**
- [ ] **Train team on emotion analysis tools**

### For Design System Managers
- [ ] **Integrate emotion design tokens**
- [ ] **Create emotion-aware component variants**
- [ ] **Set up design-development handoff process**
- [ ] **Document emotion usage guidelines**
- [ ] **Create accessibility patterns**
- [ ] **Establish version control for emotion components**

## 🔧 Troubleshooting

### Common Issues

**Emotion API Connection Issues**
```javascript
// Check API health before testing
const checkAPIHealth = async () => {
  try {
    const response = await fetch('http://localhost:8000/health');
    const health = await response.json();
    console.log('API Status:', health.status);
    return health.status === 'healthy';
  } catch (error) {
    console.error('API Health Check Failed:', error);
    return false;
  }
};
```

**Design Token Integration Problems**
```javascript
// Validate emotion design tokens
const validateEmotionTokens = (tokens) => {
  const requiredEmotions = ['happy', 'sad', 'excited', 'calm', 'frustrated'];
  const missing = requiredEmotions.filter(emotion => !tokens[emotion]);
  
  if (missing.length > 0) {
    console.warn('Missing emotion tokens:', missing);
    return false;
  }
  
  return true;
};
```

**Accessibility Compliance Issues**
```javascript
// Test accessibility features
const testAccessibility = (element) => {
  const issues = [];
  
  // Check ARIA labels
  if (!element.getAttribute('aria-label')) {
    issues.push('Missing aria-label');
  }
  
  // Check color contrast
  const style = window.getComputedStyle(element);
  const backgroundColor = style.backgroundColor;
  const color = style.color;
  
  // Add contrast checking logic here
  
  return issues;
};
```

## 📚 Additional Resources

### Design System Documentation
- [Emotion Design Tokens Reference](./Design-System-Guide.md)
- [Component Library Documentation](./Component-Library.md)
- [Accessibility Guidelines](./Accessibility-Guide.md)

### Research Methodologies
- [User Testing Protocols](./User-Testing-Guide.md)
- [A/B Testing Framework](./AB-Testing-Guide.md)
- [Analytics Implementation](./Analytics-Guide.md)

### Integration Examples
- [Figma Plugin Development](./Figma-Integration.md)
- [Sketch Integration](./Sketch-Integration.md)
- [Prototyping Tools](./Prototyping-Guide.md)

---

*This guide provides comprehensive tools and methodologies for UX teams to integrate SAMO Brain's emotion detection capabilities into their design processes, user research, and prototyping workflows.* 