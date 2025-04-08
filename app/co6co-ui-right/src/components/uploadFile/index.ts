import { withInstall } from 'co6co';

import uploadFile from './src/uploadFile';
export const UploadFile = withInstall(uploadFile);
export default UploadFile;

export * from './src';
