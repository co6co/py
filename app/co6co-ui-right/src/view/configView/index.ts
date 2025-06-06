import { withInstall } from 'co6co';

import configView, { ViewFeatures } from './src/configView';
export const ConfigView = withInstall(configView, { features: ViewFeatures });
//export default ConfigView;
export * from './src';
