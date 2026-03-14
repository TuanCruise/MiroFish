import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { translate } from 'bing-translate-api';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const zhPattern = /[\u4e00-\u9fff]+[^\x00-\x7F]*[\u4e00-\u9fff]*|[\u4e00-\u9fff]+/g;

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

async function translateBatch(texts) {
    if (texts.length === 0) return [];
    
    // Join texts with a resilient separator that Google Translate usually preserves
    const sep = "\n|xxx|\n";
    const joinedText = texts.join(sep);
    
    for (let i = 0; i < 3; i++) {
        try {
            await sleep(2000); // 2-second delay between batches
            const res = await translate(joinedText, null, 'en', true);
            let translatedParts = res.translation.split(/\n?\|?xxx\|?\n?/i).map(t => t.trim());
            
            // If split counts don't match, we might have lost a separator. Fallback to individual translation
            if (translatedParts.length !== texts.length) {
                console.log(`Separator lost during batch (Expected ${texts.length}, got ${translatedParts.length}). Falling back to individual...`);
                throw new Error("Batch length mismatch");
            }
            return translatedParts;
        } catch (e) {
            console.error(`Batch attempt ${i + 1} failed: ${e.message}`);
            await sleep(5000);
        }
    }
    
    // If all batch attempts fail, try individually
    console.log("Falling back to individual translation for this batch...");
    const results = [];
    for (const text of texts) {
        let success = false;
        for(let i=0; i<3; i++) {
            try {
                await sleep(1000);
                const res = await translate(text, null, 'en', true);
                results.push(res.translation || text);
                success = true;
                break;
            } catch(e) {
                await sleep(2000);
            }
        }
        if (!success) results.push(text);
    }
    return results;
}

async function processFile(filepath) {
    let content = fs.readFileSync(filepath, 'utf-8');
    
    const matchArray = content.match(zhPattern) || [];
    const matches = [...new Set(matchArray)];
    
    if (matches.length === 0) return false;
    
    console.log(`\nTranslating ${path.basename(filepath)} (${matches.length} phrases)...`);
    
    // Sort by length descending
    matches.sort((a, b) => b.length - a.length);
    
    const BATCH_SIZE = 15;
    const translatedMap = {};
    
    for (let i = 0; i < matches.length; i += BATCH_SIZE) {
        const batch = matches.slice(i, i + BATCH_SIZE);
        // Only translate phrases that have actual chinese
        const toTranslate = batch.filter(text => /[\u4e00-\u9fff]/.test(text));
        
        if (toTranslate.length > 0) {
            process.stdout.write(`Batch ${Math.floor(i/BATCH_SIZE)+1}/${Math.ceil(matches.length/BATCH_SIZE)}... `);
            const enTexts = await translateBatch(toTranslate);
            
            for (let j = 0; j < toTranslate.length; j++) {
                translatedMap[toTranslate[j]] = enTexts[j] || toTranslate[j];
            }
            console.log("Done");
        }
    }
    
    let newContent = content;
    for (const zhText of matches) {
        const enText = translatedMap[zhText];
        if (enText && enText !== zhText) {
            newContent = newContent.split(zhText).join(enText);
        }
    }
    
    if (newContent !== content) {
        fs.writeFileSync(filepath, newContent, 'utf-8');
        return true;
    }
    return false;
}

async function main() {
    const srcDir = path.join(__dirname, 'src');
    
    function getAllFiles(dirPath, arrayOfFiles) {
        const files = fs.readdirSync(dirPath);
        arrayOfFiles = arrayOfFiles || [];
        
        files.forEach(function(file) {
            if (fs.statSync(dirPath + "/" + file).isDirectory()) {
                arrayOfFiles = getAllFiles(dirPath + "/" + file, arrayOfFiles);
            } else {
                arrayOfFiles.push(path.join(dirPath, "/", file));
            }
        });
        return arrayOfFiles;
    }
    
    const allFiles = getAllFiles(srcDir);
    const targetFiles = allFiles.filter(f => {
        return (f.endsWith('.vue') || f.endsWith('.js')) && 
               !f.includes('locales') && 
               !f.includes('i18n.js') &&
               !f.includes('Home.vue'); 
    });
    
    let translatedCount = 0;
    for (const file of targetFiles) {
        if (await processFile(file)) {
            translatedCount++;
            console.log(`Finished: ${path.basename(file)}`);
        }
    }
    console.log(`\nDone! Translated ${translatedCount} files.`);
}

main().catch(console.error);
