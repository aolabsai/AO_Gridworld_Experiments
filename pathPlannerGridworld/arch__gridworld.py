import ao_arch as ar

arch_i = [8]
arch_z = [2]
arch_c = []

connector_function = "forward_full_conn"

Arch = ar.Arch(arch_i, arch_z, arch_c, connector_function,  description="Gridworld")