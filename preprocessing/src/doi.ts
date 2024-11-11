import pdf2doi from "pdf2doi"
import { homedir } from 'os';
import { join } from 'path';
import fs from 'fs';
import { parse } from 'json2csv';

const folderDir = join(homedir(), '/Documents/Articles/used/');
const outputCsvFile = join(folderDir, 'dois.csv');

const combinedDois: string[] = [];
pdf2doi.fromFile('./src/test.pdf').then(doi => console.log(doi))
fs.readdirSync(folderDir).forEach(async (file) => {
  if (!file.endsWith('.pdf')) return

  try {
  } catch (e) {
    console.log(file)
    console.log(e)
  }
})

// const csv = parse(combinedDois)
//
// fs.writeFile(outputCsvFile, csv, (err) => {
//   if (err) {
//     console.error('Error writing CSV file:', err);
//   } else {
//     console.log('Combined CSV file has been created:', outputCsvFile);
//   }
// });
