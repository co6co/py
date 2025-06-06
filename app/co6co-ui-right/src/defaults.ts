import type { Plugin } from 'vue';
import { INSTALLED_KEY } from '@/constants';
import { views } from '@/view/views';
import { Installer } from 'co6co';

export { moduleName, version } from '../package.json';
const _view: Array<any> = [];
Object.keys(views).forEach((key) => {
	_view.push(views[key]);
});
export default Installer([..._view] as Plugin[], INSTALLED_KEY);
