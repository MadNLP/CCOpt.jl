using Serialization

function readlog(filename::AbstractString)
    iters = Vector{MadMPEC.MadNLPCIterate}()
    open(filename, "r") do iter_file
        while (!eof(iter_file))
            push!(iters, deserialize(iter_file))
        end
    end
    return iters
end
