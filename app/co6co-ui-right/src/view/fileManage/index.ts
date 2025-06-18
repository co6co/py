import { withInstall } from 'co6co';

import fileManageView, { ViewFeatures } from './src/fileManage';
import previewView from './src/previewView';

export const FileManageView = withInstall(fileManageView, {
	features: ViewFeatures,
});
export const PreviewView = withInstall(previewView);
export * from './src';
