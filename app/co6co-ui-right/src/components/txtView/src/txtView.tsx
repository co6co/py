import { defineComponent, PropType, VNode } from 'vue';
import { ElCard } from 'element-plus';

export default defineComponent({
	props: {
		title: {
			type: String,
			required: true,
		},
		content: {
			type: String as PropType<string>,
			required: true,
		},
	},
	setup(prop) {
		const rander = (): VNode => {
			return (
				<ElCard>
					{{
						header: () => (
							<div class="card-header">
								<span>{prop.title}</span>
							</div>
						),
						default: () => (
							<div style="height: 80%; overflow: auto">
								<pre>{prop.content}</pre>
							</div>
						),
					}}
				</ElCard>
			);
		};
		//真是方法
		//ctx.expose({});
		//.d.ts 中的定义
		//rander.stateHook = stateHook;
		return rander;
	}, //end setup
});
