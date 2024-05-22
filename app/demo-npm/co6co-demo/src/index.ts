// 导入单个组件
import elButton from './el-button';
import elButtonPlus from './el-button-plus';
const commonComponts: any = {};
const commonArr = import.meta.glob(['./common/*']);
Object.keys(commonArr).forEach((fileName) => {
	const name = fileName.replace(/\.\/|\.tsx/g, '');
	commonComponts[name] = commonArr[fileName] ;
});

// 以数组的结构保存组件，便于遍历
const components = [elButton, elButtonPlus, { ...commonComponts }];
// 用于按需导入
export { elButton, elButtonPlus };
// 定义 install 方法
const install = function (Vue: any) {
	if ((install as any).installed) return;
	(install as any).installed = true;
	// 遍历并注册全局组件
	components.map((component) => {
		Vue.component(component.name, component);
	});
};
if (typeof window !== 'undefined' && window.Vue) {
	install(window.Vue);
}
export default {
	install, // 导出的对象必须具备一个 install 方法
};
