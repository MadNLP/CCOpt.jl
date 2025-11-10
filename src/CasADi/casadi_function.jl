using Libdl

abstract type CasadiSparsity{T} end

struct CscSparsity{T} <: CasadiSparsity{T}
    nrow::T
    ncol::T
    nnz::T
    colind::Vector{T}
    rows::Vector{T}
end

struct DenseSparsity{T} <: CasadiSparsity{T}
    nrow::T
    ncol::T
end

mutable struct CasadiFunction{T, IT}
    const lib::Any # Library
    const name::Symbol
    const _incref::Ptr{Cvoid}
    const _decref::Ptr{Cvoid}
    const _n_in::Ptr{Cvoid}
    const _n_out::Ptr{Cvoid}
    const _name_in::Ptr{Cvoid}
    const _name_out::Ptr{Cvoid}
    const _sparsity_in::Ptr{Cvoid}
    const _sparsity_out::Ptr{Cvoid}
    const _checkout::Ptr{Cvoid}
    const _release::Ptr{Cvoid}
    const _alloc_mem::Ptr{Cvoid}
    const _init_mem::Ptr{Cvoid}
    const _free_mem::Ptr{Cvoid}
    const _work::Ptr{Cvoid}
    const _eval::Ptr{Cvoid}

    const arg::Vector{T}
    const res::Vector{T}
    const iw::Vector{IT}
    const w::Vector{T}

    const sz_arg::IT
    const sz_res::IT
    const sz_iw::IT
    const sz_w::IT
    const n_in::IT
    const n_out::IT

    const in_sparsities::Vector{CasadiSparsity{IT}}
    const out_sparsities::Vector{CasadiSparsity{IT}}

    function CasadiFunction(libpath::String, name::Symbol)
        lib = Libdl.dlopen(libpath)
        _incref = Libdl.dlsym(lib, Symbol(name, :_incref))
        _decref = Libdl.dlsym(lib, Symbol(name, :_decref))
        _n_in = Libdl.dlsym(lib, Symbol(name, :_n_in))
        _n_out = Libdl.dlsym(lib, Symbol(name, :_n_out))
        _name_in = Libdl.dlsym(lib, Symbol(name, :_name_in))
        _name_out = Libdl.dlsym(lib, Symbol(name, :_name_out))
        _sparsity_in = Libdl.dlsym(lib, Symbol(name, :_sparsity_in))
        _sparsity_out = Libdl.dlsym(lib, Symbol(name, :_sparsity_out))
        _checkout = Libdl.dlsym(lib, Symbol(name, :_checkout))
        _release = Libdl.dlsym(lib, Symbol(name, :_release))
        _alloc_mem = Libdl.dlsym(lib, Symbol(name, :_alloc_mem))
        _init_mem = Libdl.dlsym(lib, Symbol(name, :_init_mem))
        _free_mem = Libdl.dlsym(lib, Symbol(name, :_free_mem))
        _work = Libdl.dlsym(lib, Symbol(name, :_work))
        _eval = Libdl.dlsym(lib, name)

        # get n_in and n_out
        n_in = @ccall $_n_in()::Clong
        n_out = @ccall $_n_out()::Clong

        # get work sizes
        sz_arg = Vector{Clong}(undef, 1)
        sz_res = Vector{Clong}(undef, 1)
        sz_iw = Vector{Clong}(undef, 1)
        sz_w = Vector{Clong}(undef, 1)
        err = @ccall $_work(
            pointer(sz_arg)::Ptr{Clong},
            pointer(sz_res)::Ptr{Clong},
            pointer(sz_iw)::Ptr{Clong},
            pointer(sz_w)::Ptr{Clong},
        )::Clong
        if err != 0
            error("Casadi work failed")
        end

        # allocate work vectors:
        arg = Vector{Cdouble}(undef, sz_arg[1])
        res = Vector{Cdouble}(undef, sz_res[1])
        iw = Vector{Clong}(undef, sz_iw[1])
        w = Vector{Cdouble}(undef, sz_w[1])

        # get input sparsities
        in_sparsities = Vector{CasadiSparsity{Clong}}()
        for ii in Clong(0):(n_in-Clong(1))
            sp_in = @ccall $_sparsity_in(ii::Clong)::Ptr{Clong}
            sp_in_vec = unsafe_wrap(Vector{Clong}, sp_in, (3,))
            nrow = sp_in_vec[1]
            ncol = sp_in_vec[2]
            dense = sp_in_vec[3]
            println(sp_in_vec)
            if dense != 0
                push!(in_sparsities, DenseSparsity(nrow, ncol))
            else
                colind = Vector{Clong}(undef, ncol)
                sp_in_vec = unsafe_wrap(Vector{Clong}, sp_in, (2+ncol,))
                colind .= sp_in_vec[2:end]
                nnz=colind[end]
                rows = Vector{Clong}(undef, nnz)
                sp_in_vec = unsafe_wrap(Vector{Clong}, sp_in, (2+ncol+nnz,))
                rows .= sp_in_vec[(3+ncol):end]
                push!(in_sparsities, CscSparsity(nrow, ncol, nnz, colind, rows))
            end
        end

        # get output sparsities
        out_sparsities = Vector{CasadiSparsity{Clong}}()
        for ii in Clong(0):(n_out-Clong(1))
            sp_out = @ccall $_sparsity_out(ii::Clong)::Ptr{Clong}
            sp_out_vec = unsafe_wrap(Vector{Clong}, sp_out, (3,))
            println(sp_out_vec)
            nrow = sp_out_vec[1]
            ncol = sp_out_vec[2]
            dense = sp_out_vec[3]
            if dense != 0
                push!(out_sparsities, DenseSparsity(nrow, ncol))
            else
                colind = Vector{Clong}(undef, ncol)
                sp_out_vec = unsafe_wrap(Vector{Clong}, sp_out, (2+ncol,))
                colind .= sp_out_vec[2:end]
                nnz=colind[end]
                rows = Vector{Clong}(undef, nnz)
                sp_out_vec = unsafe_wrap(Vector{Clong}, sp_out, (2+ncol+nnz,))
                rows .= sp_out_vec[(3+ncol):end]
                push!(out_sparsities, CscSparsity(nrow, ncol, nnz, colind, rows))
            end
        end

        # checkout a copy of the function and initialize the memory
        @ccall $_incref()::Cvoid
        ret = @ccall $_checkout()::Clong

        if ret != 0
            error("failed to checkout memory")
        end

        casadi_fun = new{Cdouble, Clong}(
            lib,
            name,
            _incref,
            _decref,
            _n_in,
            _n_out,
            _name_in,
            _name_out,
            _sparsity_in,
            _sparsity_out,
            _checkout,
            _release,
            _alloc_mem,
            _init_mem,
            _free_mem,
            _work,
            _eval,
            arg,
            res,
            iw,
            w,
            sz_arg[1],
            sz_res[1],
            sz_iw[1],
            sz_w[1],
            n_in,
            n_out,
            in_sparsities,
            out_sparsities,
        )

        # Setup finalizer
        function f(cf)
            _release = cf._release
            _decref = cf._decref
            @ccall $_release()::Cvoid
            @ccall $_decref()::Cvoid
        end
        finalizer(f, casadi_fun)
        return casadi_fun
    end
end
