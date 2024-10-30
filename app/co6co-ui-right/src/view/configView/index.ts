import { withInstall } from 'co6co';

import configView from './src/configView';

export const ConfigView = withInstall(configView);
export default ConfigView;

export * from './src';
