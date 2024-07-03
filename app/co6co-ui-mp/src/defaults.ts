import { makeInstaller } from './make-installer';

import { ModifyMenu } from './components';

import { WxMenuView } from './view';

import type { Plugin } from 'vue';
const components = [ModifyMenu, WxMenuView] as Plugin[];
export default makeInstaller(components);
