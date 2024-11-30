"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const fs_1 = require("fs");
const os_1 = require("os");
const path_1 = require("path");
const dayjs_1 = __importDefault(require("dayjs"));
const customParseFormat_1 = __importDefault(require("dayjs/plugin/customParseFormat"));
dayjs_1.default.extend(customParseFormat_1.default);
const isReverse = false;
if (!isReverse) {
    const rawFolderDir = (0, path_1.join)((0, os_1.homedir)(), '/Downloads/Data/months/');
    const processedFolderDir = (0, path_1.join)((0, os_1.homedir)(), '/Downloads/Data/months_processed/');
    const filePrefix = 'Plant Report_';
    if (!(0, fs_1.existsSync)(processedFolderDir))
        (0, fs_1.mkdirSync)(processedFolderDir);
    const rawFiles = (0, fs_1.readdirSync)(rawFolderDir, { withFileTypes: true });
    rawFiles.forEach(file => {
        if (file.name === '.DS_Store')
            return;
        const fileDate = file.name.split(filePrefix)[1].split('.xlsx')[0];
        const formatedDate = (0, dayjs_1.default)(fileDate, 'DD-MM-YYYY').format('YYYY-MM-DD');
        (0, fs_1.renameSync)(rawFolderDir.concat(file.name), processedFolderDir.concat(formatedDate.concat('.xlsx')));
    });
}
else {
    const rawFolderDir = (0, path_1.join)((0, os_1.homedir)(), '/Downloads/Data/months_processed/');
    const processedFolderDir = (0, path_1.join)((0, os_1.homedir)(), '/Downloads/Data/months/');
    const filePrefix = 'Plant Report_';
    if (!(0, fs_1.existsSync)(processedFolderDir))
        (0, fs_1.mkdirSync)(processedFolderDir);
    const rawFiles = (0, fs_1.readdirSync)(rawFolderDir, { withFileTypes: true });
    rawFiles.forEach(file => {
        if (file.name === '.DS_Store')
            return;
        const fileDate = file.name.split('.xlsx')[0];
        const formatedDate = (0, dayjs_1.default)(fileDate, 'MM-DD-YYYY').format('DD-MM-YYYY');
        (0, fs_1.renameSync)(rawFolderDir.concat(file.name), processedFolderDir.concat(filePrefix.concat(formatedDate.concat('.xlsx'))));
    });
}
