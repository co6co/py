<template>
	<!-- 弹出框 -->
	<el-dialog
		:title="form.title"
		v-model="form.dialogVisible"
		style="width: 50%;">
		<el-form label-width="70px"
			ref="dialogForm"
			:rules="rules"
			:model="form.fromData"
			>
			<el-form-item label="用户名" prop="userName">
				<el-input v-model="form.fromData.userName"></el-input>
			</el-form-item>
			<el-form-item v-if="form.operation===0" label="密码" prop="password">
				<el-input
					v-model="form.fromData.password"
					type="password"
					show-password>
				</el-input>
			</el-form-item>

			<el-form-item label="用户组" prop="userGroupId">
				<el-select
					style="width: 160px" 
					class="mr10"
					clearable
					v-model="form.fromData.userGroupId"
					placeholder="请选择">
					<el-option label="选择用户组" value="" />
					<el-option
						v-for="item in GroupCategoryRef.List"
						:key="item.id"
						:label="item.name"
						:value="item.id" />
				</el-select>
			</el-form-item>
			<el-form-item label="用户状态" prop="state">
				<el-select v-model="form.fromData.state" placeholder="请选择"> 
					<el-option
						v-for="item in state.list"
						:key="item.key"
						:label="item.label"
						:value="item.value" />
				</el-select>
			</el-form-item>
		</el-form>
		<template #footer>
			<span class="dialog-footer">
				<el-button @click="form.dialogVisible = false">关闭</el-button>
				<el-button @click="onDialogSave(dialogForm)">保存</el-button>
			</span>
		</template>
	</el-dialog>
</template>

<script setup lang="ts">
	import { ref, reactive, PropType } from 'vue';
	import {
		ElMessage,
		ElMessageBox,
		FormRules,
		FormInstance,
	} from 'element-plus';
	import { Plus, Minus } from '@element-plus/icons-vue';
	import * as api from '../../../api/user';
	import * as g_api from '../../../api/group';
    import {state} from './model'
	interface FromData {
		id: number;
		userName: String;
		password: String;
		userGroupId?: Number;
		state: Number;
	}
	interface dialogDataType {
		dialogVisible: boolean;
		operation: 0 | 1 | number;
		title?: string;
		id?: number;
		fromData: FromData;
	}
	const emits = defineEmits(['saved']);
	const dialogForm = ref<FormInstance>();
	const valid_userName = (rule: any, value: string, back: Function) => {
		if (value.length < 3)
			return (rule.message = '长度应该大于3'), back(new Error(rule.message));
		api.exist_svc(value,form.id).then((res) => {
			if (res.code == 0) return (rule.message = res.message), back(new Error(rule.message));
			else return back();
		});
	};
	const rules: FormRules = {
		userName: [
			{
				required: true,
				validator: valid_userName,
				message: '请输入用户名',
				trigger: ['blur'],
			},
		],
		password: [
			{
				required: true,
				min: 6,
				max: 20,
				message: '请输入6-20位密码',
				trigger: 'blur',
			},
		],
		userGroupId: [
			{
				required: true,
				message: '请选择所属组',
				trigger: ['blur', 'change'],
			},
		],
		state: [
			{
				required: true,
				message: '请选择用户状态',
				trigger: ['blur', 'change'],
			},
		],
	};

	
	let form = reactive<dialogDataType>({
		dialogVisible: false,
		operation: 0,
		id: 0, 
		fromData: {
			id: -1,
			userName: '',
			password: '',
			userGroupId: -1,
			state: 0,
		},
	});
	interface GroupCategory {
		List: Array<{ id: number; name: string }>;
	}
	const GroupCategoryRef = ref<GroupCategory>({ List: [] });
	const getGroupType = async () => {
		const res = await g_api.select_svc();
		if (res.code == 0) {
			GroupCategoryRef.value.List = res.data;
		}
	};
	getGroupType();
	const onOpenDialog = (operation: 0 | 1, item: FromData) => {
		form.dialogVisible = true;

		form.dialogVisible = true;
		form.operation = operation;
		form.id = undefined;
		switch (operation) {
			case 0:
				form.title = '增加';
				form.fromData.userName = '';
				form.fromData.password = '';
				form.fromData.userGroupId = undefined
				form.fromData.state =state.list[0].value;
				break;
			case 1:
				if (item && item.id) {
					const row = item;
					form.id = item.id;
					form.title = '编辑';
					form.fromData.userName = item.userName;
					form.fromData.password = item.password;
					form.fromData.userGroupId = item.userGroupId;
					form.fromData.state = item.state;
				}
				break;
		}
	};

	const onDialogSave = (formEl: FormInstance | undefined) => {
		if (!formEl) return;
		formEl.validate((value) => {
			if (value) {
				if (form.operation == 0) {
					api.add_svc(form.fromData).then((res) => {
						if (res.code == 0) {
							form.dialogVisible = false;
							ElMessage.success(`增加成功`);
							emits('saved');
						} else {
							ElMessage.error(`增加失败:${res.message}`);
						}
					});
				} else {
					if (form.id==undefined) {
						ElMessage.success("编辑用户Id不存在");
						return
					}
					api.edit_svc(form.id, form.fromData).then((res) => {
						if (res.code == 0) {
							form.dialogVisible = false;
							ElMessage.success(`编辑成功`);
							emits('saved');
						} else {
							ElMessage.error(`编辑失败:${res.message}`);
						}
					});
				}
			} else {
				ElMessage.error('请检查输入的数据！');
				return false;
			}
		});
	};
	defineExpose({
		onOpenDialog 
	});
</script>
