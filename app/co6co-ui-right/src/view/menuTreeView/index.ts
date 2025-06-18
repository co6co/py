import { withInstall } from 'co6co';

import menuTreeView, { ViewFeatures } from './src/menuTreeView';

export const MenuTreeView = withInstall(menuTreeView, {
	features: ViewFeatures,
});
//export default MenuTreeView;

export * from './src';
