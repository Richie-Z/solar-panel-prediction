"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const pdf2doi_1 = __importDefault(require("pdf2doi"));
const os_1 = require("os");
const path_1 = require("path");
const fs_1 = __importDefault(require("fs"));
const folderDir = (0, path_1.join)((0, os_1.homedir)(), '/Documents/Articles/used/');
const outputCsvFile = (0, path_1.join)(folderDir, 'dois.csv');
const combinedDois = [];
pdf2doi_1.default.fromFile('./src/test.pdf').then(doi => console.log(doi));
fs_1.default.readdirSync(folderDir).forEach((file) => __awaiter(void 0, void 0, void 0, function* () {
    if (!file.endsWith('.pdf'))
        return;
    try {
    }
    catch (e) {
        console.log(file);
        console.log(e);
    }
}));
// const csv = parse(combinedDois)
//
// fs.writeFile(outputCsvFile, csv, (err) => {
//   if (err) {
//     console.error('Error writing CSV file:', err);
//   } else {
//     console.log('Combined CSV file has been created:', outputCsvFile);
//   }
// });
