import { makeInstaller } from './make-installer';
import { views } from '@/views';
//import { ModifyMenu } from './components';

import type { Plugin } from 'vue';
const _view: Array<any> = [];
Object.keys(views).forEach((key) => {
	_view.push(views[key]);
});
const components = _view as Plugin[];
export default makeInstaller(components);
