

function f1(x, a)
    println("in f1, a= $a")
    return "done with f1"
end

function g2(x, a)
    fclose = x -> f1(x, a)
    println("in g2, a=$a")
    z = fclose(x)
    return z
end

x = 7.3
a = 2.0

g2(x, a)

# syntax for 2 arguments to fclose
function f1(x, a, b)
    println("in f1, a= $a")
    println("in f1, b= $b")
    return "done with f1"
end

function g2(x, a, b)
    fclose = x -> f1(x, a, b)
    z = fclose(x)
    return z
end

g2(3., 4., 5.)