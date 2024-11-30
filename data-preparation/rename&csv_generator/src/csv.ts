import xlsx from 'node-xlsx';
import { homedir } from 'os';
import { join } from 'path';
import fs from 'fs';
import { parse } from 'json2csv';

const folderDir = join(homedir(), '/Documents/Programming/Projects/on-going/solar-panel-prediction/raw-data/months_processed/');
const outputCsvFile = join(folderDir, 'combined_output.csv');

const combinedData: any[] = [];

fs.readdir(folderDir, (err, files) => {
  if (err) {
    console.error('Error reading directory:', err);
    return;
  }

  const xlsxFiles = files.filter(file => file.endsWith('.xlsx'));

  xlsxFiles.forEach(file => {
    const filePath = join(folderDir, file);
    const workSheetsFromBuffer = xlsx.parse(filePath);

    const sheetData = workSheetsFromBuffer[0].data;

    if (combinedData.length === 0) {
      combinedData.push(sheetData[1]);
    }

    for (let i = 2; i < sheetData.length; i++) {
      combinedData.push(sheetData[i]);
    }
  });

  const csv = parse(combinedData);

  fs.writeFile(outputCsvFile, csv, (err) => {
    if (err) {
      console.error('Error writing CSV file:', err);
    } else {
      console.log('Combined CSV file has been created:', outputCsvFile);
    }
  });
});
