import fs from 'fs';
import path from 'path';
import axios from 'axios';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const API_URL = 'https://chatbot-test.etc.vn:9006/llm/v1/chat/completions';

// Function to call LLM
async function translateVueFile(content) {
  try {
    const response = await axios.post(
      API_URL,
      {
        model: 'openai/gpt-oss-120b',
        messages: [
          {
            role: 'system',
            content: `You are an expert Vue.js developer and translator. 
Your task is to translate ALL Chinese text within the provided Vue/JS source code into English.
CRITICAL RULES:
1. Preserve all Vue template syntax, Vue directives, HTML tags, CSS styles, JavaScript logic, and imports exactly as they are.
2. DO NOT change variable names, component names, or classes.
3. Only translate the human-readable text strings (in template blocks, console logs, error messages, etc.).
4. Do not output any markdown formatting like \`\`\`vue or \`\`\`. Output ONLY the raw source code.
5. Do not add any conversational text or explanations.`
          },
          {
            role: 'user',
            content
          }
        ]
      },
      {
        headers: {
          'Content-Type': 'application/json'
        },
        httpsAgent: new (await import('https')).Agent({  
          rejectUnauthorized: false
        })
      }
    );

    if (!response.data || !response.data.choices) {
      console.error('API returned unexpected success response:', JSON.stringify(response.data).substring(0, 500));
      return null;
    }

    const translated = response.data.choices[0].message.content;
    // Remove markdown code blocks if the LLM still adds them
    return translated.replace(/^```(vue|html|javascript|js)?\n/m, '').replace(/```$/m, '').trim();
  } catch (err) {
    console.error('LLM Translation error:', err.message);
    if(err.response) console.error(err.response.data);
    return null;
  }
}

// Recursively find files
function getAllFiles(dirPath, arrayOfFiles) {
  const files = fs.readdirSync(dirPath);

  arrayOfFiles = arrayOfFiles || [];

  files.forEach(function(file) {
    if (fs.statSync(dirPath + "/" + file).isDirectory()) {
      arrayOfFiles = getAllFiles(dirPath + "/" + file, arrayOfFiles);
    } else {
      const ext = path.extname(file);
      if (ext === '.vue' || ext === '.js') {
        arrayOfFiles.push(path.join(dirPath, "/", file));
      }
    }
  });

  return arrayOfFiles;
}

const targetDir = path.join(__dirname, 'src');
const files = getAllFiles(targetDir);

// Simple check for Chinese characters
const containsChinese = (str) => /[\u4e00-\u9fa5]/.test(str);

async function main() {
  for (const file of files) {
    // Skip locales setup files we made since they are already translated/setup manually
    if (file.includes('locales') || file.includes('i18n.js') || file.includes('Home.vue')) {
      continue;
    }

    const content = fs.readFileSync(file, 'utf8');
    
    if (containsChinese(content)) {
      console.log(`Translating: ${file}`);
      const translatedContent = await translateVueFile(content);
      
      if (translatedContent && !containsChinese(translatedContent)) {
        fs.writeFileSync(file, translatedContent, 'utf8');
        console.log(`✅ Success: ${file}`);
      } else if (translatedContent) {
        fs.writeFileSync(file, translatedContent, 'utf8');
        console.log(`⚠️ Partial/Incomplete Translation: ${file}`);
      } else {
        console.log(`❌ Failed: ${file}`);
      }
    }
  }
  console.log('All files processed.');
}

main();
