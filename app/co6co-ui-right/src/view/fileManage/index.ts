import { withInstall } from 'co6co';

import fileManageView from './src/fileManage';
import previewView from './src/previewView';

export const FileManageView = withInstall(fileManageView);
export const PreviewView = withInstall(previewView);
export * from './src';
