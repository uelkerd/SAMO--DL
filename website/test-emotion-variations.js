// Test script to verify emotion variations are working
console.log('🧪 Testing emotion variations...');

// Simulate the generateEmotionVariations function
function generateEmotionVariations(primaryEmotion) {
    console.log('🎭 Generating emotion variations for:', primaryEmotion.emotion);
    
    // Define emotion relationships and variations
    const emotionRelations = {
        'excitement': [
            { emotion: 'joy', confidence: primaryEmotion.confidence * 0.8 },
            { emotion: 'optimism', confidence: primaryEmotion.confidence * 0.6 },
            { emotion: 'gratitude', confidence: primaryEmotion.confidence * 0.4 },
            { emotion: 'neutral', confidence: 0.15 }
        ],
        'disappointment': [
            { emotion: 'sadness', confidence: primaryEmotion.confidence * 0.7 },
            { emotion: 'frustration', confidence: primaryEmotion.confidence * 0.5 },
            { emotion: 'anxiety', confidence: primaryEmotion.confidence * 0.3 },
            { emotion: 'neutral', confidence: 0.2 }
        ],
        'sadness': [
            { emotion: 'disappointment', confidence: primaryEmotion.confidence * 0.7 },
            { emotion: 'anxiety', confidence: primaryEmotion.confidence * 0.4 },
            { emotion: 'frustration', confidence: primaryEmotion.confidence * 0.3 },
            { emotion: 'neutral', confidence: 0.2 }
        ]
    };
    
    // Get variations for the primary emotion, or use default if not found
    const variations = emotionRelations[primaryEmotion.emotion] || [
        { emotion: 'neutral', confidence: 0.2 },
        { emotion: 'optimism', confidence: 0.3 },
        { emotion: 'gratitude', confidence: 0.25 },
        { emotion: 'calm', confidence: 0.15 }
    ];
    
    // Add some randomness to make it more realistic
    return variations.map(variation => ({
        emotion: variation.emotion,
        confidence: Math.max(0.05, Math.min(0.95, variation.confidence + (Math.random() - 0.5) * 0.1))
    }));
}

// Test with different primary emotions
const testCases = [
    { emotion: 'excitement', confidence: 0.739 },
    { emotion: 'disappointment', confidence: 0.327 },
    { emotion: 'sadness', confidence: 0.887 }
];

console.log('🧪 Testing emotion variations for different inputs:');
testCases.forEach((primaryEmotion, index) => {
    console.log(`\n📝 Test ${index + 1}: Primary emotion = ${primaryEmotion.emotion} (${Math.round(primaryEmotion.confidence * 100)}%)`);
    
    const variations = generateEmotionVariations(primaryEmotion);
    const allEmotions = [primaryEmotion, ...variations];
    
    // Sort by confidence
    allEmotions.sort((a, b) => b.confidence - a.confidence);
    
    console.log('🎭 Generated emotions:');
    allEmotions.forEach((emotion, i) => {
        console.log(`  ${i + 1}. ${emotion.emotion}: ${Math.round(emotion.confidence * 100)}%`);
    });
});

console.log('\n✅ Emotion variations test completed!');
