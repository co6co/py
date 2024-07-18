import { withInstall } from 'co6co';

import configView from './src/configView.vue';

export const ConfigView = withInstall(configView);
export default ConfigView;

export * from './src';
