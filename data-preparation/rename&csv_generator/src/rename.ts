import { existsSync, mkdirSync, readdirSync, renameSync } from "fs";
import { homedir } from "os";
import { join } from "path";
import dayjs from 'dayjs'
import customParseFormat from 'dayjs/plugin/customParseFormat'

dayjs.extend(customParseFormat)

const isReverse = false;

if (!isReverse) {
  const rawFolderDir = join(homedir(), '/Downloads/Data/months/');
  const processedFolderDir = join(homedir(), '/Downloads/Data/months_processed/');
  const filePrefix = 'Plant Report_'

  if (!existsSync(processedFolderDir)) mkdirSync(processedFolderDir);
  const rawFiles = readdirSync(rawFolderDir, { withFileTypes: true })

  rawFiles.forEach(file => {
    if (file.name === '.DS_Store') return
    const fileDate = file.name.split(filePrefix)[1].split('.xlsx')[0]
    const formatedDate = dayjs(fileDate, 'DD-MM-YYYY').format('YYYY-MM-DD')

    renameSync(rawFolderDir.concat(file.name), processedFolderDir.concat(formatedDate.concat('.xlsx')))
  })
} else {
  const rawFolderDir = join(homedir(), '/Downloads/Data/months_processed/');
  const processedFolderDir = join(homedir(), '/Downloads/Data/months/');
  const filePrefix = 'Plant Report_'

  if (!existsSync(processedFolderDir)) mkdirSync(processedFolderDir);
  const rawFiles = readdirSync(rawFolderDir, { withFileTypes: true })

  rawFiles.forEach(file => {
    if (file.name === '.DS_Store') return
    const fileDate = file.name.split('.xlsx')[0]
    const formatedDate = dayjs(fileDate, 'MM-DD-YYYY').format('DD-MM-YYYY')

    renameSync(rawFolderDir.concat(file.name), processedFolderDir.concat(filePrefix.concat(formatedDate.concat('.xlsx'))))
  })
}
