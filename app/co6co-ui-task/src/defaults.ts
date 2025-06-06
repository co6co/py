import { INSTALLED_KEY } from './constants';

import { Installer } from 'co6co';
import { views } from '@/views';

import type { Plugin } from 'vue';
const _view: Array<any> = [];
Object.keys(views).forEach((key) => {
	_view.push(views[key]);
});
//import type { App, Plugin } from '@vue/runtime-core';
export default Installer([..._view] as Plugin[], INSTALLED_KEY);
