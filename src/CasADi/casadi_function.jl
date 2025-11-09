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

struct CasadiFunction
    lib::Any # Library
    name::Symbol
    _incref::Symbol
    _decref::Symbol
    _n_in::Symbol
    _n_out::Symbol
    _name_in::Symbol
    _name_out::Symbol
    _sparsity_in::Symbol
    _sparsity_out::Symbol
    _checkout::Symbol
    _release::Symbol
    _alloc_mem::Symbol
    _init_mem::Symbol
    _free_mem::Symbol
    _work::Symbol
    _eval::Symbol

    mem_ptr::Vector{Cvoid}

    arg::Vector{Cdouble}
    res::Vector{Cdouble}
    iw::Vector{Cint}
    w::Vector{Cdouble}    

    sz_arg::Vector{Cint}
    sz_res::Vector{Cint}
    sz_iw::Vector{Cint}
    sz_w::Vector{Cint}
    n_in::Cint
    n_out::Cint

    in_sparsities::Vector{CasadiSparsity{Cint}}
    out_sparsities::Vector{CasadiSparsity{Cint}}
    
    function CasadiFunction(libpath::String, name::Symbol)
        _lib = Libdl.dlopen(libpath)
        _incref = Libdl.dlsym(lib, Symbol(name,:_incref))
        _decref = Libdl.dlsym(lib, Symbol(name,:_decref))
        _n_in = Libdl.dlsym(lib, Symbol(name,:_n_in))
        _n_out = Libdl.dlsym(lib, Symbol(name,:_n_out))
        _name_in = Libdl.dlsym(lib, Symbol(name,:_name_in))
        _name_out = Libdl.dlsym(lib, Symbol(name,:_name_out))
        _sparsity_in = Libdl.dlsym(lib, Symbol(name,:_sparsity_in))
        _sparsity_out = Libdl.dlsym(lib, Symbol(name,:_sparsity_out))
        _checkout = Libdl.dlsym(lib, Symbol(name,:_checkout))
        _release = Libdl.dlsym(lib, Symbol(name,:_release))
        _alloc_mem = Libdl.dlsym(lib, Symbol(name,:_alloc_mem))
        _init_mem = Libdl.dlsym(lib, Symbol(name,:_init_mem))
        _free_mem = Libdl.dlsym(lib, Symbol(name,:_free_mem))
        _work = Libdl.dlsym(lib, Symbol(name,:_work))
        _eval = Libdl.dlsym(lib, name)

        # get n_in and n_out
        n_in = @ccall $_n_in()::Cint
        n_out = @ccall $_n_out()::Cint
        
        # get work sizes
        err = @ccall $_work(pointer(sz_arg), pointer(sz_res), pointer(sz_iw), pointer(sz_w))::Cint
        if err
            error("Casadi work failed")
        end

        in_sparsities = Vector{CasadiSparsity{Cint}}()
        for ii=Cint(0):(n_in-Cint(1))
            sp_in = @ccall $_sparsity_in(ii)::Ptr{Cint}
            sp_in_vec = unsafe_wrap(Vector{Cint}, sp_in, (3,))
            nrow = sp_in[1]
            ncol = sp_in[2]
            dense = sp_in_vec[3]
            if dense
                push!(in_sparsities, DenseSparsity(nrow,ncol))
            else
                colind = Vector{Cint}(undef, ncol)
                unsafe_copyto!(colind, sp_in+2, ncol)
                nnz=colind[end]
                rows = Vector{Cint}(undef, nnz)
                unsafe_copyto!(colind, sp_in+ncol+3, nnz)
                push!(in_sparsities, CscSparsity(nrow, ncol, nnz, colind, rows))
            end
        end

        out_sparsities = Vector{CasadiSparsity{Cint}}()
        for ii=Cint(0):(n_out-Cint(1))
            sp_out = @ccall $_sparsity_out(ii)::Ptr{Cint}
            sp_out_vec = unsafe_wrap(Vector{Cint}, sp_out, (3,))
            nrow = sp_out_vec[1]
            ncol = sp_out_vec[2]
            dense = sp_out_vec[3]
            if dense
                push!(out_sparsities, DenseSparsity(nrow,ncol))
            else
                colind = Vector{Cint}(undef, ncol)
                unsafe_copyto!(colind, sp_out+2, ncol)
                nnz=colind[end]
                rows = Vector{Cint}(undef, nnz)
                unsafe_copyto!(colind, sp_out+ncol+3, nnz)
                push!(out_sparsities, CscSparsity(nrow, ncol, nnz, colind, rows))
            end
        end

        out_sparsities = Vector{CscSparsity{Cint}}()
    end
end
