# 🤗 Hugging Face AI Text Generation Setup

## **FREE AI-Powered Sample Text Generation**

The SAMO-DL demo now uses **Hugging Face Inference API** to generate dynamic, AI-powered sample texts instead of static ones!

## **🚀 Quick Setup (2 minutes)**

### **Step 1: Get Your FREE API Token**

1. **Sign up** at [huggingface.co](https://huggingface.co/)
2. **Go to Settings** → **Access Tokens**
3. **Create New Token** → Name it "SAMO-DL-Demo"
4. **Copy the token** (starts with `hf_`)

### **Step 2: Add Token to Demo**

1. **Open** `website/js/simple-demo-functions.js`
2. **Find line 45**: `'Authorization': 'Bearer hf_your_token_here',`
3. **Replace** `hf_your_token_here` with your actual token
4. **Save** the file

### **Step 3: Test It!**

1. **Open** the demo: `http://localhost:8081/comprehensive-demo.html`
2. **Click "Generate"** button
3. **Watch** AI generate unique journal text every time!

## **🎯 What You Get**

- ✅ **1,000 FREE requests/month** (more than enough for demos)
- ✅ **5 different emotional prompts** (excitement, anxiety, mixed, calm, motivation)
- ✅ **Dynamic text generation** (never the same text twice!)
- ✅ **Automatic fallback** to static samples if API fails
- ✅ **Visual feedback** with loading states and animations

## **🔧 API Details**

- **Model**: GPT-2 (free, no credit card required)
- **Endpoint**: `https://api-inference.huggingface.co/models/gpt2`
- **Parameters**: 
  - `max_length: 150` (perfect paragraph length)
  - `temperature: 0.8` (creative but coherent)
  - `top_p: 0.9` (balanced creativity)

## **🎨 Visual Feedback**

- **Purple border**: AI is generating text
- **Green border**: AI text successfully generated
- **Orange border**: Fallback to static samples

## **🛡️ Security Note**

- **Never commit** your API token to version control
- **Keep it private** - this is your personal token
- **Free tier** is sufficient for demos and testing

## **🚨 Troubleshooting**

**If Generate button shows static text:**
1. Check your API token is correct
2. Verify you're signed in to Hugging Face
3. Check browser console for error messages
4. Ensure you have internet connection

**If you see "Generating AI text..." forever:**
- The API might be loading the model (first request takes longer)
- Wait 10-15 seconds, then try again
- Check your internet connection

---

**🎉 Enjoy your AI-powered demo!** Every click of "Generate" creates unique, emotional journal text perfect for testing the SAMO-DL emotion analysis pipeline!
