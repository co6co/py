export enum ViewFeature {
	//** 查看 */
	view = 'view',
	//** 查看 */
	get = 'get',
	//** 增加 */
	add = 'add',
	//** 编辑 */
	edit = 'edit',
	//** 删除 */
	del = 'del',
	//** 设备 */
	setting = 'setting',
	//** 检查 */
	check = 'check',
	//** 下载 */
	downloads = 'downloads',
	//** 下载 */
	download = 'download',
	upload = 'upload',
	//** 关联 */
	associated = 'associated',
	//** 重置 */
	reset = 'reset',
	//** 推送 */
	push = 'push',
	//** 使生效 **/
	effective = 'effective',
	settingName = 'settingName',
	//** 设备编号 */
	settingNo = 'settingNo',
	//** 设置优先级 */
	settingPriority = 'settingPriority',
	sched = 'sched',
	stop = 'stop',
	execute = 'execute',
}
export const ViewFeatureDesc = {
	view: '查看',
	get: '获取',
	add: '增加',
	edit: '编辑',
	del: '删除',
	setting: '设备',
	check: '检查',
	downloads: '下载',
	download: '下载',
	upload: '上传',
	associated: '关联',
	reset: '重置',
	push: '推送',
	effective: '使生效',
	settingName: '设置名称',
	settingNo: '设备编号',
	settingPriority: '设置优先级',
	sched: '调度',
	stop: '停止',
	execute: '执行',
	/**
	 *
	 * @param field 字段名称
	 * @param feature 所有字段 {get:"get",add:"add" sched:{value:"sched",text:"调度表"}}
	 * @returns
	 */
	getDesc: function (field: string, feature: object) {
		//使用自定义属性描述
		if (
			feature.hasOwnProperty(field) &&
			typeof feature[field] == 'object' &&
			feature[field].hasOwnProperty('text')
		)
			return feature[field]['text'];
		//使用本对象描述
		if (this.hasOwnProperty(field)) return this[field];
		//使用 穿入值
		return field;
	},
};
