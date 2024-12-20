import { withInstall } from 'co6co';

import dragUploader from './src/dragUploader';

export const DragUploader = withInstall(dragUploader);
export default DragUploader;
export * from './src';
