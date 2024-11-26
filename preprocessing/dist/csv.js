"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const node_xlsx_1 = __importDefault(require("node-xlsx"));
const os_1 = require("os");
const path_1 = require("path");
const fs_1 = __importDefault(require("fs"));
const json2csv_1 = require("json2csv");
const folderDir = (0, path_1.join)((0, os_1.homedir)(), '/Documents/Programming/Projects/on-going/solar-panel-prediction/raw-data/months_processed/');
const outputCsvFile = (0, path_1.join)(folderDir, 'combined_output.csv');
const combinedData = [];
fs_1.default.readdir(folderDir, (err, files) => {
    if (err) {
        console.error('Error reading directory:', err);
        return;
    }
    const xlsxFiles = files.filter(file => file.endsWith('.xlsx'));
    xlsxFiles.forEach(file => {
        const filePath = (0, path_1.join)(folderDir, file);
        const workSheetsFromBuffer = node_xlsx_1.default.parse(filePath);
        const sheetData = workSheetsFromBuffer[0].data;
        if (combinedData.length === 0) {
            combinedData.push(sheetData[1]);
        }
        for (let i = 2; i < sheetData.length; i++) {
            combinedData.push(sheetData[i]);
        }
    });
    const csv = (0, json2csv_1.parse)(combinedData);
    fs_1.default.writeFile(outputCsvFile, csv, (err) => {
        if (err) {
            console.error('Error writing CSV file:', err);
        }
        else {
            console.log('Combined CSV file has been created:', outputCsvFile);
        }
    });
});
