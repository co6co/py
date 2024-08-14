import { withInstall } from 'co6co';

import download from './src/download';

export const Download = withInstall(download);
export default Download;

export * from './src';
