import { withInstall } from 'co6co';

import dictView, { ViewFeatures } from './src/dictView';

export const DictView = withInstall(dictView, {
	features: ViewFeatures,
});
//export default DictView;

export * from './src';
