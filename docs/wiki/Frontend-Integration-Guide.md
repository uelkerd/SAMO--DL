# 🎨 Frontend Integration Guide

Welcome, Frontend Developers! This guide will help you integrate SAMO Brain's emotion detection capabilities into your web and mobile applications with beautiful, responsive UI components.

## 🚀 **Quick Start (5 minutes)**

### **1. Test the API**
```javascript
// Test emotion detection
const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ text: 'I am feeling happy today!' })
});

const result = await response.json();
console.log(`Emotion: ${result.predicted_emotion}`);
console.log(`Confidence: ${result.confidence}`);
```

### **2. Expected Response**
```json
{
  "text": "I am feeling happy today!",
  "predicted_emotion": "happy",
  "confidence": 0.964,
  "prediction_time_ms": 25.3,
  "probabilities": {
    "anxious": 0.001,
    "calm": 0.002,
    "content": 0.004,
    "excited": 0.004,
    "frustrated": 0.002,
    "grateful": 0.005,
    "happy": 0.964,
    "hopeful": 0.004,
    "overwhelmed": 0.001,
    "proud": 0.002,
    "sad": 0.008,
    "tired": 0.002
  }
}
```

---

## 🎯 **Emotion Categories & Colors**

### **Emotion Color Palette**
```javascript
const EMOTION_COLORS = {
  anxious: '#FF6B6B',      // Red - worry, nervousness
  calm: '#4ECDC4',         // Teal - peaceful, relaxed
  content: '#45B7D1',      // Blue - satisfied, pleased
  excited: '#FFA726',      // Orange - enthusiastic, thrilled
  frustrated: '#FF7043',   // Deep Orange - annoyed, irritated
  grateful: '#66BB6A',     // Green - thankful, appreciative
  happy: '#FFD54F',        // Yellow - joyful, cheerful
  hopeful: '#81C784',      // Light Green - optimistic, confident
  overwhelmed: '#9575CD',  // Purple - stressed, burdened
  proud: '#4DB6AC',        // Cyan - accomplished, confident
  sad: '#7986CB',          // Indigo - unhappy, sorrowful
  tired: '#A1887F'         // Brown - exhausted, weary
};

const EMOTION_ICONS = {
  anxious: '😰',
  calm: '😌',
  content: '😊',
  excited: '🤩',
  frustrated: '😤',
  grateful: '🙏',
  happy: '😄',
  hopeful: '🤗',
  overwhelmed: '😵',
  proud: '😎',
  sad: '😢',
  tired: '😴'
};
```

---

## 🔌 **Integration Examples**

### **React Component**

```jsx
import React, { useState, useEffect } from 'react';
import './EmotionAnalyzer.css';

const EMOTION_COLORS = {
  anxious: '#FF6B6B', calm: '#4ECDC4', content: '#45B7D1',
  excited: '#FFA726', frustrated: '#FF7043', grateful: '#66BB6A',
  happy: '#FFD54F', hopeful: '#81C784', overwhelmed: '#9575CD',
  proud: '#4DB6AC', sad: '#7986CB', tired: '#A1887F'
};

const EMOTION_ICONS = {
  anxious: '😰', calm: '😌', content: '😊', excited: '🤩',
  frustrated: '😤', grateful: '🙏', happy: '😄', hopeful: '🤗',
  overwhelmed: '😵', proud: '😎', sad: '😢', tired: '😴'
};

const EmotionAnalyzer = () => {
  const [text, setText] = useState('');
  const [emotion, setEmotion] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const analyzeEmotion = async () => {
    if (!text.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setEmotion(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    analyzeEmotion();
  };

  return (
    <div className="emotion-analyzer">
      <h2>🧠 SAMO Brain - Emotion Analyzer</h2>
      
      <form onSubmit={handleSubmit} className="analyzer-form">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="How are you feeling today? Share your thoughts..."
          className="emotion-input"
          rows={4}
        />
        
        <button 
          type="submit" 
          disabled={loading || !text.trim()}
          className="analyze-button"
        >
          {loading ? 'Analyzing...' : 'Analyze Emotion'}
        </button>
      </form>

      {error && (
        <div className="error-message">
          ❌ Error: {error}
        </div>
      )}

      {emotion && (
        <div className="emotion-result">
          <div 
            className="emotion-card"
            style={{ 
              backgroundColor: EMOTION_COLORS[emotion.predicted_emotion] + '20',
              borderColor: EMOTION_COLORS[emotion.predicted_emotion]
            }}
          >
            <div className="emotion-header">
              <span className="emotion-icon">
                {EMOTION_ICONS[emotion.predicted_emotion]}
              </span>
              <h3 className="emotion-name">
                {emotion.predicted_emotion.charAt(0).toUpperCase() + 
                 emotion.predicted_emotion.slice(1)}
              </h3>
            </div>
            
            <div className="confidence-bar">
              <div 
                className="confidence-fill"
                style={{ 
                  width: `${emotion.confidence * 100}%`,
                  backgroundColor: EMOTION_COLORS[emotion.predicted_emotion]
                }}
              />
            </div>
            
            <p className="confidence-text">
              Confidence: {(emotion.confidence * 100).toFixed(1)}%
            </p>
            
            <div className="emotion-probabilities">
              <h4>All Emotions:</h4>
              <div className="probability-grid">
                {Object.entries(emotion.probabilities)
                  .sort(([,a], [,b]) => b - a)
                  .map(([emotionName, probability]) => (
                    <div 
                      key={emotionName}
                      className="probability-item"
                      style={{ 
                        backgroundColor: EMOTION_COLORS[emotionName] + '20',
                        borderColor: EMOTION_COLORS[emotionName]
                      }}
                    >
                      <span className="prob-icon">{EMOTION_ICONS[emotionName]}</span>
                      <span className="prob-name">{emotionName}</span>
                      <span className="prob-value">{(probability * 100).toFixed(1)}%</span>
                    </div>
                  ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default EmotionAnalyzer;
```

### **CSS Styling**

```css
.emotion-analyzer {
  max-width: 600px;
  margin: 0 auto;
  padding: 20px;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.analyzer-form {
  margin-bottom: 20px;
}

.emotion-input {
  width: 100%;
  padding: 15px;
  border: 2px solid #e0e0e0;
  border-radius: 8px;
  font-size: 16px;
  resize: vertical;
  transition: border-color 0.3s ease;
}

.emotion-input:focus {
  outline: none;
  border-color: #4ECDC4;
  box-shadow: 0 0 0 3px rgba(78, 205, 196, 0.1);
}

.analyze-button {
  width: 100%;
  padding: 12px 24px;
  background: linear-gradient(135deg, #4ECDC4, #45B7D1);
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 16px;
  font-weight: 600;
  cursor: pointer;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
  margin-top: 10px;
}

.analyze-button:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(78, 205, 196, 0.3);
}

.analyze-button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.error-message {
  padding: 12px;
  background-color: #ffebee;
  color: #c62828;
  border-radius: 8px;
  margin-bottom: 20px;
  border-left: 4px solid #c62828;
}

.emotion-result {
  animation: slideIn 0.3s ease;
}

.emotion-card {
  padding: 20px;
  border-radius: 12px;
  border: 2px solid;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
}

.emotion-header {
  display: flex;
  align-items: center;
  margin-bottom: 15px;
}

.emotion-icon {
  font-size: 2.5rem;
  margin-right: 15px;
}

.emotion-name {
  margin: 0;
  font-size: 1.8rem;
  font-weight: 700;
  color: #333;
}

.confidence-bar {
  height: 8px;
  background-color: #f0f0f0;
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 10px;
}

.confidence-fill {
  height: 100%;
  transition: width 0.8s ease;
}

.confidence-text {
  margin: 0 0 20px 0;
  font-size: 14px;
  color: #666;
  text-align: center;
}

.emotion-probabilities h4 {
  margin: 0 0 15px 0;
  color: #333;
}

.probability-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 8px;
}

.probability-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 8px;
  border-radius: 6px;
  border: 1px solid;
  font-size: 12px;
}

.prob-icon {
  font-size: 1.2rem;
  margin-bottom: 4px;
}

.prob-name {
  font-weight: 600;
  margin-bottom: 2px;
  text-transform: capitalize;
}

.prob-value {
  font-size: 11px;
  opacity: 0.8;
}

@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@media (max-width: 480px) {
  .emotion-analyzer {
    padding: 15px;
  }
  
  .probability-grid {
    grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
  }
}
```

### **Vue.js Component**

```vue
<template>
  <div class="emotion-analyzer">
    <h2>🧠 SAMO Brain - Emotion Analyzer</h2>
    
    <form @submit.prevent="analyzeEmotion" class="analyzer-form">
      <textarea
        v-model="text"
        placeholder="How are you feeling today? Share your thoughts..."
        class="emotion-input"
        rows="4"
      />
      
      <button 
        type="submit" 
        :disabled="loading || !text.trim()"
        class="analyze-button"
      >
        {{ loading ? 'Analyzing...' : 'Analyze Emotion' }}
      </button>
    </form>

    <div v-if="error" class="error-message">
      ❌ Error: {{ error }}
    </div>

    <div v-if="emotion" class="emotion-result">
      <div 
        class="emotion-card"
        :style="{ 
          backgroundColor: emotionColors[emotion.predicted_emotion] + '20',
          borderColor: emotionColors[emotion.predicted_emotion]
        }"
      >
        <div class="emotion-header">
          <span class="emotion-icon">
            {{ emotionIcons[emotion.predicted_emotion] }}
          </span>
          <h3 class="emotion-name">
            {{ capitalizeFirst(emotion.predicted_emotion) }}
          </h3>
        </div>
        
        <div class="confidence-bar">
          <div 
            class="confidence-fill"
            :style="{ 
              width: `${emotion.confidence * 100}%`,
              backgroundColor: emotionColors[emotion.predicted_emotion]
            }"
          />
        </div>
        
        <p class="confidence-text">
          Confidence: {{ (emotion.confidence * 100).toFixed(1) }}%
        </p>
        
        <div class="emotion-probabilities">
          <h4>All Emotions:</h4>
          <div class="probability-grid">
            <div 
              v-for="[emotionName, probability] in sortedProbabilities"
              :key="emotionName"
              class="probability-item"
              :style="{ 
                backgroundColor: emotionColors[emotionName] + '20',
                borderColor: emotionColors[emotionName]
              }"
            >
              <span class="prob-icon">{{ emotionIcons[emotionName] }}</span>
              <span class="prob-name">{{ emotionName }}</span>
              <span class="prob-value">{{ (probability * 100).toFixed(1) }}%</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'EmotionAnalyzer',
  data() {
    return {
      text: '',
      emotion: null,
      loading: false,
      error: null,
      emotionColors: {
        anxious: '#FF6B6B', calm: '#4ECDC4', content: '#45B7D1',
        excited: '#FFA726', frustrated: '#FF7043', grateful: '#66BB6A',
        happy: '#FFD54F', hopeful: '#81C784', overwhelmed: '#9575CD',
        proud: '#4DB6AC', sad: '#7986CB', tired: '#A1887F'
      },
      emotionIcons: {
        anxious: '😰', calm: '😌', content: '😊', excited: '🤩',
        frustrated: '😤', grateful: '🙏', happy: '😄', hopeful: '🤗',
        overwhelmed: '😵', proud: '😎', sad: '😢', tired: '😴'
      }
    };
  },
  computed: {
    sortedProbabilities() {
      if (!this.emotion) return [];
      return Object.entries(this.emotion.probabilities)
        .sort(([,a], [,b]) => b - a);
    }
  },
  methods: {
    async analyzeEmotion() {
      if (!this.text.trim()) return;

      this.loading = true;
      this.error = null;

      try {
        const response = await fetch('http://localhost:8000/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: this.text })
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        this.emotion = await response.json();
      } catch (err) {
        this.error = err.message;
      } finally {
        this.loading = false;
      }
    },
    capitalizeFirst(str) {
      return str.charAt(0).toUpperCase() + str.slice(1);
    }
  }
};
</script>

<style scoped>
/* Same CSS as React component above */
</style>
```

### **Mobile App (React Native)**

```jsx
import React, { useState } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  ScrollView,
  StyleSheet,
  Alert,
  ActivityIndicator
} from 'react-native';

const EMOTION_COLORS = {
  anxious: '#FF6B6B', calm: '#4ECDC4', content: '#45B7D1',
  excited: '#FFA726', frustrated: '#FF7043', grateful: '#66BB6A',
  happy: '#FFD54F', hopeful: '#81C784', overwhelmed: '#9575CD',
  proud: '#4DB6AC', sad: '#7986CB', tired: '#A1887F'
};

const EMOTION_ICONS = {
  anxious: '😰', calm: '😌', content: '😊', excited: '🤩',
  frustrated: '😤', grateful: '🙏', happy: '😄', hopeful: '🤗',
  overwhelmed: '😵', proud: '😎', sad: '😢', tired: '😴'
};

const EmotionAnalyzer = () => {
  const [text, setText] = useState('');
  const [emotion, setEmotion] = useState(null);
  const [loading, setLoading] = useState(false);

  const analyzeEmotion = async () => {
    if (!text.trim()) return;

    setLoading(true);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setEmotion(result);
    } catch (error) {
      Alert.alert('Error', error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.title}>🧠 SAMO Brain</Text>
      <Text style={styles.subtitle}>Emotion Analyzer</Text>

      <View style={styles.form}>
        <TextInput
          style={styles.input}
          value={text}
          onChangeText={setText}
          placeholder="How are you feeling today? Share your thoughts..."
          multiline
          numberOfLines={4}
          textAlignVertical="top"
        />

        <TouchableOpacity
          style={[styles.button, (!text.trim() || loading) && styles.buttonDisabled]}
          onPress={analyzeEmotion}
          disabled={!text.trim() || loading}
        >
          {loading ? (
            <ActivityIndicator color="white" />
          ) : (
            <Text style={styles.buttonText}>Analyze Emotion</Text>
          )}
        </TouchableOpacity>
      </View>

      {emotion && (
        <View style={styles.resultContainer}>
          <View style={[
            styles.emotionCard,
            { backgroundColor: EMOTION_COLORS[emotion.predicted_emotion] + '20' }
          ]}>
            <View style={styles.emotionHeader}>
              <Text style={styles.emotionIcon}>
                {EMOTION_ICONS[emotion.predicted_emotion]}
              </Text>
              <Text style={styles.emotionName}>
                {emotion.predicted_emotion.charAt(0).toUpperCase() + 
                 emotion.predicted_emotion.slice(1)}
              </Text>
            </View>

            <View style={styles.confidenceBar}>
              <View
                style={[
                  styles.confidenceFill,
                  {
                    width: `${emotion.confidence * 100}%`,
                    backgroundColor: EMOTION_COLORS[emotion.predicted_emotion]
                  }
                ]}
              />
            </View>

            <Text style={styles.confidenceText}>
              Confidence: {(emotion.confidence * 100).toFixed(1)}%
            </Text>
          </View>
        </View>
      )}
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
    padding: 20,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 5,
  },
  subtitle: {
    fontSize: 18,
    color: '#666',
    textAlign: 'center',
    marginBottom: 30,
  },
  form: {
    marginBottom: 20,
  },
  input: {
    backgroundColor: 'white',
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    padding: 15,
    fontSize: 16,
    marginBottom: 15,
    minHeight: 100,
  },
  button: {
    backgroundColor: '#4ECDC4',
    padding: 15,
    borderRadius: 8,
    alignItems: 'center',
  },
  buttonDisabled: {
    opacity: 0.6,
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  resultContainer: {
    marginTop: 20,
  },
  emotionCard: {
    padding: 20,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: '#4ECDC4',
  },
  emotionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 15,
  },
  emotionIcon: {
    fontSize: 40,
    marginRight: 15,
  },
  emotionName: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
  },
  confidenceBar: {
    height: 8,
    backgroundColor: '#f0f0f0',
    borderRadius: 4,
    overflow: 'hidden',
    marginBottom: 10,
  },
  confidenceFill: {
    height: '100%',
  },
  confidenceText: {
    textAlign: 'center',
    fontSize: 14,
    color: '#666',
  },
});

export default EmotionAnalyzer;
```

---

## 🎨 **UI/UX Best Practices**

### **Real-time Emotion Display**

```jsx
const RealTimeEmotionDisplay = ({ emotion, confidence }) => {
  const [isAnimating, setIsAnimating] = useState(false);

  useEffect(() => {
    if (emotion) {
      setIsAnimating(true);
      setTimeout(() => setIsAnimating(false), 1000);
    }
  }, [emotion]);

  return (
    <div className={`emotion-display ${isAnimating ? 'pulse' : ''}`}>
      <div className="emotion-badge">
        <span className="emotion-icon">{EMOTION_ICONS[emotion]}</span>
        <span className="emotion-label">{emotion}</span>
        <div className="confidence-indicator">
          <div 
            className="confidence-bar"
            style={{ width: `${confidence * 100}%` }}
          />
        </div>
      </div>
    </div>
  );
};
```

### **Emotion Trend Visualization**

```jsx
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const EmotionTrendChart = ({ emotionHistory }) => {
  const data = emotionHistory.map((entry, index) => ({
    time: index + 1,
    emotion: entry.emotion,
    confidence: entry.confidence * 100,
    timestamp: entry.timestamp
  }));

  return (
    <div className="emotion-trend">
      <h3>Emotion Trend</h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="time" />
          <YAxis domain={[0, 100]} />
          <Tooltip />
          <Line 
            type="monotone" 
            dataKey="confidence" 
            stroke="#4ECDC4" 
            strokeWidth={2}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};
```

---

## 🔄 **State Management**

### **Redux Toolkit (React)**

```javascript
// emotionSlice.js
import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';

export const analyzeEmotion = createAsyncThunk(
  'emotion/analyze',
  async (text) => {
    const response = await fetch('http://localhost:8000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });
    
    if (!response.ok) {
      throw new Error('Analysis failed');
    }
    
    return response.json();
  }
);

const emotionSlice = createSlice({
  name: 'emotion',
  initialState: {
    currentEmotion: null,
    emotionHistory: [],
    loading: false,
    error: null
  },
  reducers: {
    clearEmotion: (state) => {
      state.currentEmotion = null;
      state.error = null;
    }
  },
  extraReducers: (builder) => {
    builder
      .addCase(analyzeEmotion.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(analyzeEmotion.fulfilled, (state, action) => {
        state.loading = false;
        state.currentEmotion = action.payload;
        state.emotionHistory.push({
          ...action.payload,
          timestamp: new Date().toISOString()
        });
      })
      .addCase(analyzeEmotion.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message;
      });
  }
});

export const { clearEmotion } = emotionSlice.actions;
export default emotionSlice.reducer;
```

### **Vuex (Vue.js)**

```javascript
// store/modules/emotion.js
export default {
  namespaced: true,
  state: {
    currentEmotion: null,
    emotionHistory: [],
    loading: false,
    error: null
  },
  mutations: {
    SET_LOADING(state, loading) {
      state.loading = loading;
    },
    SET_EMOTION(state, emotion) {
      state.currentEmotion = emotion;
      state.emotionHistory.push({
        ...emotion,
        timestamp: new Date().toISOString()
      });
    },
    SET_ERROR(state, error) {
      state.error = error;
    },
    CLEAR_EMOTION(state) {
      state.currentEmotion = null;
      state.error = null;
    }
  },
  actions: {
    async analyzeEmotion({ commit }, text) {
      commit('SET_LOADING', true);
      commit('SET_ERROR', null);

      try {
        const response = await fetch('http://localhost:8000/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text })
        });

        if (!response.ok) {
          throw new Error('Analysis failed');
        }

        const emotion = await response.json();
        commit('SET_EMOTION', emotion);
      } catch (error) {
        commit('SET_ERROR', error.message);
      } finally {
        commit('SET_LOADING', false);
      }
    }
  }
};
```

---

## 🎯 **Performance Optimization**

### **Debounced Input**

```jsx
import { useCallback } from 'react';
import { debounce } from 'lodash';

const DebouncedEmotionAnalyzer = () => {
  const [text, setText] = useState('');
  const [emotion, setEmotion] = useState(null);

  const debouncedAnalyze = useCallback(
    debounce(async (text) => {
      if (!text.trim()) return;

      try {
        const response = await fetch('http://localhost:8000/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text })
        });

        if (response.ok) {
          const result = await response.json();
          setEmotion(result);
        }
      } catch (error) {
        console.error('Analysis failed:', error);
      }
    }, 500),
    []
  );

  const handleTextChange = (newText) => {
    setText(newText);
    debouncedAnalyze(newText);
  };

  return (
    <div>
      <textarea
        value={text}
        onChange={(e) => handleTextChange(e.target.value)}
        placeholder="Type your thoughts..."
      />
      {emotion && (
        <div>Detected emotion: {emotion.predicted_emotion}</div>
      )}
    </div>
  );
};
```

### **Caching Strategy**

```javascript
class EmotionCache {
  constructor() {
    this.cache = new Map();
    this.maxSize = 100;
  }

  get(text) {
    const hash = this.hashText(text);
    return this.cache.get(hash);
  }

  set(text, emotion) {
    const hash = this.hashText(text);
    
    if (this.cache.size >= this.maxSize) {
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }
    
    this.cache.set(hash, {
      emotion,
      timestamp: Date.now()
    });
  }

  hashText(text) {
    return btoa(text).slice(0, 20);
  }

  clear() {
    this.cache.clear();
  }
}

const emotionCache = new EmotionCache();

const analyzeWithCache = async (text) => {
  // Check cache first
  const cached = emotionCache.get(text);
  if (cached && Date.now() - cached.timestamp < 300000) { // 5 minutes
    return cached.emotion;
  }

  // Fetch from API
  const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text })
  });

  const emotion = await response.json();
  
  // Cache the result
  emotionCache.set(text, emotion);
  
  return emotion;
};
```

---

## 🚨 **Error Handling & User Feedback**

### **Comprehensive Error Handling**

```jsx
const EmotionAnalyzerWithErrorHandling = () => {
  const [text, setText] = useState('');
  const [emotion, setEmotion] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const analyzeEmotion = async () => {
    if (!text.trim()) {
      setError('Please enter some text to analyze');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });

      if (response.status === 429) {
        setError('Rate limit exceeded. Please wait a moment and try again.');
        return;
      }

      if (response.status === 400) {
        const errorData = await response.json();
        setError(`Invalid request: ${errorData.error}`);
        return;
      }

      if (!response.ok) {
        setError(`Server error: ${response.status}`);
        return;
      }

      const result = await response.json();
      setEmotion(result);
    } catch (error) {
      if (error.name === 'TypeError' && error.message.includes('fetch')) {
        setError('Network error. Please check your connection.');
      } else {
        setError(`Unexpected error: ${error.message}`);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="emotion-analyzer">
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="How are you feeling today?"
        className={error ? 'input-error' : ''}
      />

      {error && (
        <div className="error-message">
          <span className="error-icon">⚠️</span>
          {error}
        </div>
      )}

      <button 
        onClick={analyzeEmotion}
        disabled={loading || !text.trim()}
        className="analyze-button"
      >
        {loading ? 'Analyzing...' : 'Analyze Emotion'}
      </button>

      {emotion && (
        <div className="success-message">
          <span className="success-icon">✅</span>
          Analysis complete!
        </div>
      )}
    </div>
  );
};
```

---

## 📱 **Mobile-Specific Considerations**

### **Touch-Friendly Interface**

```jsx
const MobileEmotionAnalyzer = () => {
  return (
    <div className="mobile-emotion-analyzer">
      <div className="mobile-header">
        <h1>🧠 SAMO Brain</h1>
        <p>How are you feeling?</p>
      </div>

      <div className="mobile-input-container">
        <textarea
          className="mobile-textarea"
          placeholder="Share your thoughts..."
          maxLength={500}
        />
        <div className="character-count">0/500</div>
      </div>

      <div className="mobile-emotion-buttons">
        {Object.entries(EMOTION_ICONS).map(([emotion, icon]) => (
          <button
            key={emotion}
            className="emotion-button"
            style={{ backgroundColor: EMOTION_COLORS[emotion] }}
          >
            <span className="button-icon">{icon}</span>
            <span className="button-text">{emotion}</span>
          </button>
        ))}
      </div>

      <button className="mobile-analyze-button">
        Analyze My Emotions
      </button>
    </div>
  );
};
```

---

## 📞 **Support & Resources**

- **API Documentation**: [Complete API Reference](API-Reference)
- **Design System**: [Emotion Design Components](Design-System-Guide)
- **Performance**: [Frontend Performance Guide](Performance-Guide)
- **GitHub Issues**: [Report Issues](https://github.com/your-org/SAMO--DL/issues)

---

**Ready to build beautiful emotion-aware interfaces?** Start with the [Quick Start](#-quick-start-5-minutes) section above! 🎨✨ 