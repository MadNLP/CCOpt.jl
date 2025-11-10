using Libdl

abstract type CasADiSparsity{T} end

struct CscSparsity{T} <: CasADiSparsity{T}
    nrow::T
    ncol::T
    nnz::T
    colind::Vector{T}
    rows::Vector{T}
end

struct DenseSparsity{T} <: CasADiSparsity{T}
    nrow::T
    ncol::T
end

nnz(cs::DenseSparsity) = cs.nrow*cs.ncol
nnz(cs::CscSparsity) = cs.nnz

# TODO(@anton) Make this generic on int and float types.
#              This seems to be nontrivial because @ccall takes the typevars
#              and does not decompose them to concrete types when it can.
#              In principle one could also specialize on number of args but this is
#              perhaps not super useful as this can't be known at compile time anyway
mutable struct CasADiFunction
    const lib::Ptr{Cvoid} # Library
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

    const arg_vec::Vector{
        Union{SparseMatrixCSC{Cdouble, Clong}, Matrix{Cdouble}, Vector{Cdouble}},
    }
    const res_vec::Vector{
        Union{SparseMatrixCSC{Cdouble, Clong}, Matrix{Cdouble}, Vector{Cdouble}},
    }
    const iw_vec::Vector{Clong}
    const w_vec::Vector{Cdouble}

    const arg_ptr_vec::Vector{Ptr{Cdouble}}
    const res_ptr_vec::Vector{Ptr{Cdouble}}

    const sz_arg::Clong
    const sz_res::Clong
    const sz_iw::Clong
    const sz_w::Clong
    const n_in::Clong
    const n_out::Clong

    const in_sparsities::Vector{CasADiSparsity{Clong}}
    const out_sparsities::Vector{CasADiSparsity{Clong}}

    function CasADiFunction(libpath::String, name::Symbol)
        lib = Libdl.dlopen(libpath)
        return CasADiFunction(lib, name)
    end

    function CasADiFunction(lib::Ptr{Cvoid}, name::Symbol)
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
            error("CasADi work failed")
        end

        arg_vec = Vector{
            Union{SparseMatrixCSC{Cdouble, Clong}, Matrix{Cdouble}, Vector{Cdouble}},
        }()
        arg_ptr_vec = Vector{Ptr{Cdouble}}()

        # get input sparsities
        in_sparsities = Vector{CasADiSparsity{Clong}}()
        for ii in Clong(0):(n_in-Clong(1))
            sp_in = @ccall $_sparsity_in(ii::Clong)::Ptr{Clong}
            sp_in_vec = unsafe_wrap(Vector{Clong}, sp_in, (3,))
            nrow = sp_in_vec[1]
            ncol = sp_in_vec[2]
            dense = sp_in_vec[3]
            if dense != 0
                push!(in_sparsities, DenseSparsity(nrow, ncol))
                if ncol <= 1
                    push!(arg_vec, Vector{Cdouble}(undef, nnz(in_sparsities[end])))
                else
                    push!(arg_vec, Matrix{Cdouble}(undef, nrow, ncol))
                end
                push!(arg_ptr_vec, pointer(arg_vec[end]))
            else
                colind = Vector{Clong}(undef, ncol)
                sp_in_vec = unsafe_wrap(Vector{Clong}, sp_in, (3+ncol,))
                colind .= sp_in_vec[3:end] .+ 1
                nnz_ = colind[end]-1
                rows = Vector{Clong}(undef, nnz)
                sp_in_vec = unsafe_wrap(Vector{Clong}, sp_in, (3+ncol+nnz_,))
                rows .= sp_in_vec[(4+ncol):end] .+ 1
                push!(in_sparsities, CscSparsity(nrow, ncol, nnz_, colind, rows))
                nzval = Vector{Cdouble}(undef, nnz_)
                sparsein = SparseMatrixCSC{Cdouble, Clong}(nrow, ncol, colind, rows, nzval)
                push!(arg_vec, sparsein)
                push!(arg_ptr_vec, pointer(nzval))
            end
        end

        res_vec = Vector{
            Union{SparseMatrixCSC{Cdouble, Clong}, Matrix{Cdouble}, Vector{Cdouble}},
        }()
        res_ptr_vec = Vector{Ptr{Cdouble}}()
        # get output sparsities
        out_sparsities = Vector{CasADiSparsity{Clong}}()
        for ii in Clong(0):(n_out-Clong(1))
            sp_out = @ccall $_sparsity_out(ii::Clong)::Ptr{Clong}
            sp_out_vec = unsafe_wrap(Vector{Clong}, sp_out, (3,))
            nrow = sp_out_vec[1]
            ncol = sp_out_vec[2]
            dense = sp_out_vec[3]
            if dense != 0
                push!(out_sparsities, DenseSparsity(nrow, ncol))
                if ncol == 1
                    push!(res_vec, Vector{Cdouble}(undef, nnz(out_sparsities[end])))
                else
                    push!(res_vec, Matrix{Cdouble}(undef, nrow, ncol))
                end
                push!(res_ptr_vec, pointer(res_vec[end]))
            else
                colind = Vector{Clong}(undef, ncol+1)
                sp_out_vec = unsafe_wrap(Vector{Clong}, sp_out, (3+ncol,))
                colind .= sp_out_vec[3:end] .+ 1
                nnz_ = colind[end] - 1
                rows = Vector{Clong}(undef, nnz_)
                sp_out_vec = unsafe_wrap(Vector{Clong}, sp_out, (3+ncol+nnz_,))
                rows .= sp_out_vec[(4+ncol):end] .+ 1
                push!(out_sparsities, CscSparsity(nrow, ncol, nnz_, colind, rows))
                nzval = Vector{Cdouble}(undef, nnz_)
                sparseout = SparseMatrixCSC{Cdouble, Clong}(nrow, ncol, colind, rows, nzval)
                push!(res_vec, sparseout)
                push!(res_ptr_vec, pointer(nzval))
            end
        end

        # allocate work vectors:
        iw_vec = Vector{Clong}(undef, sz_iw[1])
        w_vec = Vector{Cdouble}(undef, sz_w[1])

        # checkout a copy of the function and initialize the memory
        @ccall $_incref()::Cvoid
        ret = @ccall $_checkout()::Clong

        if ret != 0
            error("failed to checkout memory")
        end

        casadi_fun = new(
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
            arg_vec,
            res_vec,
            iw_vec,
            w_vec,
            arg_ptr_vec,
            res_ptr_vec,
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

check_arg_type(arg::Any, inarg::Any) = false
check_arg_type(arg::T, inarg::T) where {T <: Vector} = true
check_arg_type(arg::T, inarg::T) where {T <: Matrix} = true
check_arg_type(arg::T, inarg::T) where {T <: SparseMatrixCSC} = true

check_arg_size(arg::Any, inarg::Any) = false
check_arg_size(arg::T, inarg::T) where {T <: Vector} = length(arg) == length(inarg)
check_arg_size(arg::T, inarg::T) where {T <: Matrix} = size(arg) == size(inarg)

function check_arg_size(arg::T, inarg::T) where {T <: SparseMatrixCSC}
    len_compat = size(arg) == size(inarg)
    return len_compat &&
           all(rowvals(arg) .== rowvals(inarg)) &&
           all(arg.colptr .== arg.colptr)
end

function check_arg(fun::CasADiFunction, ii::Int, inarg::Any)
    if !check_arg_type(fun.arg_vec[ii], inarg)
        error(
            "CasADi function $(fun.name) expected input argument $(ii) of type $(typeof(fun.arg_vec[ii]))",
        )
    end
    if !check_arg_size(fun.arg_vec[ii], inarg)
        error(
            "CasADi function $(fun.name) got wrong shape input argument $(ii) of type $(typeof(fun.arg_vec[ii]))",
        )
    end
    return nothing
end

function (fun::CasADiFunction)(args...)
    # Check number of args
    if length(args) != fun.n_in
        error(
            "Wrong number of args passed to CasADi function $(fun.name). Expected $(fun.n_in)",
        )
    end

    # Get eval fpointer
    eval = fun._eval

    # Copy to arguments
    for ii in 1:fun.n_in
        check_arg(fun, ii, args[ii])
        fun.arg_vec[ii] .= args[ii]
    end

    ret = @ccall $eval(
        pointer(fun.arg_ptr_vec)::Ptr{Ptr{Cdouble}},
        pointer(fun.res_ptr_vec)::Ptr{Ptr{Cdouble}},
        pointer(fun.iw_vec)::Ptr{Clong},
        pointer(fun.w_vec)::Ptr{Cdouble},
    )::Clong

    if ret != 0
        error("Error evaluating function $(fun.name)")
    end

    return fun.res_vec
end
