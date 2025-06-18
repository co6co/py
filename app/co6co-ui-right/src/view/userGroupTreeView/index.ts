import { withInstall } from 'co6co';
import userGroupTreeView, { ViewFeatures } from './src/userGroupTreeView';

export const UserGroupTreeView = withInstall(userGroupTreeView, {
	features: ViewFeatures,
});
//export default UserGroupTreeView;
export * from './src';
