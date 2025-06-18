import { withInstall } from 'co6co';

import dictTypeView, { ViewFeatures } from './src/dictTypeView';

export const DictTypeView = withInstall(dictTypeView, {
	features: ViewFeatures,
});
//export default DictTypeView;

export * from './src';
